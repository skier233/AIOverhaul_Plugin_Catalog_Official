"""Deep Taste recommender — advanced multi-signal recommendation engine.

Builds on top of the content-based system with additional intelligence:

1. **Multi-embedding fusion** — Separate weights for visual-style (DINOv3),
   visual-semantic (MetaCLIP2), and audio embeddings instead of a single
   embedding knob.  DINOv3 excels at visual-to-visual style matching while
   MetaCLIP2 captures semantic scene content.
2. **Cluster-aware diversification** — Uses precomputed content clusters to
   ensure results span different content niches rather than all coming from
   the same cluster.  Optionally allows drilling into a specific cluster.
3. **Face-based performer discovery** — Finds scenes featuring faces
   visually similar to performers the user has engaged with, even when
   those scenes lack performer tags in Stash.
4. **Configurable tag exclusion / performer pinning** via the built-in
   config field system (``type: "tags"`` / ``type: "performers"``).

Each signal contributes to a [0, 1] composite score.  Full debug metadata
is attached for transparency.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Set, Tuple

import numpy as np
import sqlalchemy as sa
from sqlalchemy import select

from stash_ai_server.db.session import get_session_local
from stash_ai_server.models.entity_embeddings import EntityEmbedding
from stash_ai_server.models.taste_profile import (
    ContentCluster,
    ContentClusterMember,
    TasteCentroid,
)
from stash_ai_server.recommendations.engagement.scorer import (
    score_all_watched_scenes,
)
from stash_ai_server.recommendations.models import RecContext, RecommendationRequest
from stash_ai_server.recommendations.registry import recommender
from stash_ai_server.recommendations.utils.scene_fetch import fetch_scenes_by_ids
from stash_ai_server.recommendations.utils.stash_tags import (
    build_tfidf_vectors,
    build_user_performer_profile,
    build_user_tag_profile,
    compute_document_frequencies,
    compute_idf,
    cosine_similarity,
    fetch_all_scene_tag_ids,
    fetch_scene_tag_ids,
    fetch_tag_names,
    resolve_blacklisted_tag_ids,
    score_scene_against_profile,
)
from stash_ai_server.recommendations.utils.watch_history import (
    load_watch_history_summary,
)
from stash_ai_server.recommendations.utils.embedding_similarity import (
    VISUAL_PREFIX,
    VISUAL_DINOV3_PREFIX,
    AUDIO_PREFIXES,
    _fetch_embeddings_by_prefix,
    find_similar_by_cached_centroids,
    find_similar_by_seed_embeddings,
    find_similar_by_taste_centroid,
    _distance_to_similarity,
)
from stash_ai_server.recommendations.utils.ai_tag_similarity import (
    compute_ai_tag_similarity,
)
from stash_ai_server.recommendations.utils.taste_weighting import (
    build_taste_weights,
    compute_data_depth,
    DEFAULT_HALF_LIFE_DAYS,
)
from stash_ai_server.recommendations.utils.negative_signals import (
    build_negative_performer_profile,
    build_negative_tag_profile,
    compute_combined_negative_penalty,
    compute_negative_detail,
    split_by_engagement,
)
from stash_ai_server.db.embedding_store import find_similar_entities

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TAG_WEIGHT = 0.5
DEFAULT_PERFORMER_WEIGHT = 0.3
DEFAULT_STUDIO_WEIGHT = 0.05
DEFAULT_DINOV3_WEIGHT = 0.3       # visual-to-visual style
DEFAULT_METACLIP2_WEIGHT = 0.2    # visual-semantic
DEFAULT_AUDIO_WEIGHT = 0.1
DEFAULT_AI_TAG_WEIGHT = 0.3
DEFAULT_FACE_WEIGHT = 0.15
DEFAULT_NEGATIVE_WEIGHT = 0.15
DEFAULT_RECENCY_HALF_LIFE = DEFAULT_HALF_LIFE_DAYS
DEFAULT_HISTORY_LIMIT = 400
DEFAULT_MIN_WATCH_SECONDS = 15.0
DEFAULT_CANDIDATE_LIMIT = 400
DEFAULT_CLUSTER_DIVERSITY = 0.3   # 0 = off, 1 = full diversity reranking
DEFAULT_TOP_TAG_DEBUG = 8


def _coerce_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_scene_performers(payloads: Mapping[int, Dict[str, Any]]) -> Dict[int, Set[int]]:
    result: Dict[int, Set[int]] = {}
    for scene_id, payload in payloads.items():
        pids: Set[int] = set()
        for p in payload.get("performers", []):
            pid = p.get("id")
            if pid is not None:
                try:
                    pids.add(int(pid))
                except (TypeError, ValueError):
                    continue
        if pids:
            result[scene_id] = pids
    return result


def _extract_scene_studio_ids(payloads: Mapping[int, Dict[str, Any]]) -> Dict[int, int]:
    result: Dict[int, int] = {}
    for scene_id, payload in payloads.items():
        studio = payload.get("studio")
        if isinstance(studio, dict):
            sid = studio.get("id")
            if sid is not None:
                try:
                    result[scene_id] = int(sid)
                except (TypeError, ValueError):
                    continue
    return result


def _build_studio_affinity(
    studio_map: Mapping[int, int],
    engagement_scores: Mapping[int, float] | None,
) -> Dict[int, float]:
    raw: Dict[int, float] = defaultdict(float)
    for scene_id, studio_id in studio_map.items():
        weight = 1.0
        if engagement_scores:
            weight = engagement_scores.get(scene_id, 0.5)
        raw[studio_id] += weight
    if not raw:
        return {}
    mx = max(raw.values())
    return {sid: v / mx for sid, v in raw.items()} if mx > 0 else {}


def _annotate_top_tags(
    tags: Set[int],
    profile: Mapping[int, float],
    idf: Mapping[int, float],
    tag_names: Mapping[int, str],
    limit: int = DEFAULT_TOP_TAG_DEBUG,
) -> List[Dict[str, Any]]:
    scored = []
    for tid in tags:
        p_w = profile.get(tid, 0.0)
        i_w = idf.get(tid, 0.0)
        scored.append((tid, p_w, i_w))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        {"tag_id": tid, "tag_name": tag_names.get(tid, f"tag_{tid}"),
         "profile_weight": round(pw, 4), "idf": round(iw, 4)}
        for tid, pw, iw in scored[:limit]
    ]


def _compute_tag_contributions(
    scene_vector: Mapping[int, float],
    user_profile: Mapping[int, float],
    tag_names: Mapping[int, str],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return top tags driving cosine similarity between scene and profile."""
    contribs = []
    for tid in scene_vector:
        if tid in user_profile:
            c = scene_vector[tid] * user_profile[tid]
            contribs.append({
                "tag_id": tid,
                "tag_name": tag_names.get(tid, f"tag_{tid}"),
                "contribution": round(c, 6),
                "scene_w": round(scene_vector[tid], 4),
                "profile_w": round(user_profile[tid], 4),
            })
    contribs.sort(key=lambda x: x["contribution"], reverse=True)
    return contribs[:limit]


def _build_performer_detail(
    scene_performers: Set[int] | None,
    performer_profile: Mapping[int, float],
    payloads: Mapping[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per-performer affinity detail for debug."""
    if not scene_performers or not performer_profile:
        return []
    detail = []
    for pid in scene_performers:
        affinity = performer_profile.get(pid, 0.0)
        # Try to find performer name from payloads
        name = None
        for pay in payloads.values():
            for p in pay.get("performers", []):
                if p.get("id") is not None and int(p["id"]) == pid:
                    name = p.get("name")
                    break
            if name:
                break
        detail.append({
            "performer_id": pid,
            "performer_name": name or f"performer_{pid}",
            "affinity": round(affinity, 4),
        })
    detail.sort(key=lambda x: x["affinity"], reverse=True)
    return detail


# ---------------------------------------------------------------------------
# Face-based performer similarity
# ---------------------------------------------------------------------------

def _find_scenes_by_liked_faces(
    watched_scene_ids: Set[int],
    engagement_scores: Mapping[int, float] | None = None,
    limit: int = 150,
) -> Dict[int, float]:
    """Find scenes containing faces similar to faces from liked content.

    1. Get FaceClusters linked to performers in liked watched scenes.
    2. Use their centroids to find similar face embeddings in other scenes.
    3. Return ``{scene_id: similarity}`` for unwatched scenes.
    """
    from stash_ai_server.models.detections import FaceCluster, FaceEmbedding

    # Get the top-engaged watched scene IDs for face lookup
    if engagement_scores:
        reference_ids = sorted(
            watched_scene_ids,
            key=lambda s: engagement_scores.get(s, 0.0),
            reverse=True,
        )[:50]
    else:
        reference_ids = list(watched_scene_ids)[:50]

    # Find face clusters linked to performers that appear in reference scenes
    try:
        with get_session_local()() as session:
            # Get performer IDs from the reference scenes via Stash tags
            # We use face clusters with performer_id set
            cluster_rows = session.execute(
                select(FaceCluster.id, FaceCluster.centroid, FaceCluster.performer_id)
                .where(
                    FaceCluster.status == "identified",
                    FaceCluster.centroid.isnot(None),
                    FaceCluster.performer_id.isnot(None),
                )
            ).all()

            if not cluster_rows:
                return {}

            # Also get clusters from scenes (via face embeddings) the user liked
            liked_clusters = session.execute(
                select(
                    FaceEmbedding.cluster_id,
                )
                .where(
                    FaceEmbedding.entity_type == "scene",
                    FaceEmbedding.entity_id.in_(reference_ids),
                    FaceEmbedding.cluster_id.isnot(None),
                )
                .distinct()
            ).scalars().all()

            liked_cluster_ids = set(int(c) for c in liked_clusters if c is not None)

            # Use centroids of clusters the user has engaged with
            search_centroids = []
            for row in cluster_rows:
                if row.id in liked_cluster_ids and row.centroid is not None:
                    search_centroids.append(list(row.centroid))

            if not search_centroids:
                return {}

            # Search for similar faces across all scenes
            best_sim: Dict[int, float] = {}
            for centroid_vec in search_centroids[:10]:  # limit queries
                results = session.execute(
                    select(
                        FaceEmbedding.entity_id,
                        FaceEmbedding.embedding.cosine_distance(centroid_vec).label("dist"),
                    )
                    .where(
                        FaceEmbedding.entity_type == "scene",
                    )
                    .order_by("dist")
                    .limit(limit)
                ).all()

                for r in results:
                    eid = int(r.entity_id)
                    if eid in watched_scene_ids:
                        continue
                    sim = max(0.0, 1.0 - float(r.dist))
                    if sim > best_sim.get(eid, 0.0):
                        best_sim[eid] = sim

            _log.info(
                "deep_taste face search: %d clusters searched, %d candidates",
                len(search_centroids), len(best_sim),
            )
            return best_sim

    except Exception:
        _log.debug("deep_taste face similarity unavailable", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Cluster-aware diversification
# ---------------------------------------------------------------------------

def _load_scene_clusters() -> Dict[int, int]:
    """Return ``{scene_id: cluster_id}`` from precomputed clusters."""
    try:
        with get_session_local()() as session:
            rows = session.execute(
                select(ContentClusterMember.scene_id, ContentClusterMember.cluster_id)
            ).all()
        return {int(r.scene_id): int(r.cluster_id) for r in rows}
    except Exception:
        return {}


def _load_cluster_affinities() -> Dict[int, float]:
    """Return ``{cluster_id: user_affinity}`` from ContentCluster."""
    try:
        with get_session_local()() as session:
            rows = session.execute(
                select(ContentCluster.id, ContentCluster.user_affinity)
            ).all()
        return {
            int(r.id): float(r.user_affinity) if r.user_affinity is not None else 0.5
            for r in rows
        }
    except Exception:
        return {}


def _diversify_by_cluster(
    scored: List[Tuple[int, float, Dict[str, Any]]],
    scene_clusters: Dict[int, int],
    cluster_affinities: Dict[int, float],
    diversity_factor: float = 0.3,
    target_cluster_id: int | None = None,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    """Rerank scored list to ensure cluster diversity.

    If ``target_cluster_id`` is set, boost scenes from that cluster instead
    of diversifying (drill-down mode).

    ``diversity_factor`` in [0, 1] controls how aggressively to diversify:
    0 = pure score ranking, 1 = maximally diverse.
    """
    if not scored or not scene_clusters:
        return scored

    # Drill-down mode: boost target cluster (independent of diversity_factor)
    if target_cluster_id is not None:
        boosted = []
        for sid, score, dbg in scored:
            cid = scene_clusters.get(sid)
            if cid == target_cluster_id:
                boosted.append((sid, score * 1.5, dbg))
            else:
                boosted.append((sid, score, dbg))
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    # Diversity reranking only when diversity_factor > 0
    if diversity_factor <= 0:
        return scored

    # Diversity reranking via MMR-style (maximal marginal relevance)
    remaining = list(scored)
    reranked: List[Tuple[int, float, Dict[str, Any]]] = []
    cluster_counts: Dict[int, int] = defaultdict(int)

    while remaining:
        best_idx = 0
        best_mmr = -1.0

        for i, (sid, score, _dbg) in enumerate(remaining):
            cid = scene_clusters.get(sid)
            # Cluster penalty: more scenes from same cluster → lower MMR
            cluster_penalty = 0.0
            if cid is not None:
                count = cluster_counts.get(cid, 0)
                cluster_penalty = min(1.0, count * 0.15) * diversity_factor
                # Bonus for high-affinity clusters
                affinity = cluster_affinities.get(cid, 0.5)
                cluster_penalty -= (affinity - 0.5) * 0.1 * diversity_factor

            mmr = score * (1.0 - cluster_penalty)
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i

        chosen = remaining.pop(best_idx)
        cid = scene_clusters.get(chosen[0])
        if cid is not None:
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        reranked.append(chosen)

    return reranked


# ---------------------------------------------------------------------------
# Multi-embedding similarity with separate weights
# ---------------------------------------------------------------------------

def _compute_multi_embedding_similarity(
    *,
    watched_scene_ids: Set[int],
    engagement_scores: Mapping[int, float] | None,
    exclude_scene_ids: Set[int],
    dinov3_weight: float,
    metaclip2_weight: float,
    audio_weight: float,
) -> Tuple[Dict[int, float], Dict[str, Dict[int, float]]]:
    """Compute similarity using multiple embedding types with separate weights.

    Returns ``(combined_map, per_type_maps)`` where:
    - ``combined_map``: ``{scene_id: weighted_similarity}`` blending all types
    - ``per_type_maps``: ``{"dinov3": {sid: sim}, "metaclip2": ..., "audio": ...}``
      with raw (un-weighted) similarities per embedding type.
    """
    result: Dict[int, float] = {}
    per_type: Dict[str, Dict[int, float]] = {}
    total_weight = dinov3_weight + metaclip2_weight + audio_weight
    if total_weight <= 0:
        return {}, {}

    prefixes_and_weights = []
    if dinov3_weight > 0:
        prefixes_and_weights.append((VISUAL_DINOV3_PREFIX, dinov3_weight, "dinov3"))
    if metaclip2_weight > 0:
        prefixes_and_weights.append((VISUAL_PREFIX, metaclip2_weight, "metaclip2"))
    if audio_weight > 0:
        for ap in AUDIO_PREFIXES:
            prefixes_and_weights.append((ap, audio_weight / len(AUDIO_PREFIXES), "audio"))

    for prefix, weight, type_key in prefixes_and_weights:
        # Try cached centroids first (fast, precomputed)
        sim_map: Dict[int, float] = {}
        try:
            sim_map = find_similar_by_cached_centroids(
                exclude_scene_ids=exclude_scene_ids,
                type_prefix=prefix,
            )
            if sim_map:
                _log.debug("deep_taste: cached centroid hit for %s (%d candidates)", prefix, len(sim_map))
        except Exception:
            _log.debug("deep_taste: cached centroid lookup failed for %s", prefix, exc_info=True)

        # Fallback: compute centroid on-the-fly from watched scenes
        if not sim_map:
            try:
                sim_map = find_similar_by_taste_centroid(
                    watched_scene_ids=list(watched_scene_ids),
                    engagement_scores=engagement_scores,
                    type_prefix=prefix,
                )
            except Exception:
                _log.debug("deep_taste: on-the-fly centroid failed for %s", prefix, exc_info=True)
                continue

        if sim_map:
            # Store raw per-type sims (merge audio sub-prefixes)
            if type_key not in per_type:
                per_type[type_key] = {}
            for sid, sim in sim_map.items():
                if sid in exclude_scene_ids:
                    continue
                # Keep best per-type sim
                if sim > per_type[type_key].get(sid, 0.0):
                    per_type[type_key][sid] = sim

            normalized_weight = weight / total_weight
            for sid, sim in sim_map.items():
                if sid in exclude_scene_ids:
                    continue
                result[sid] = result.get(sid, 0.0) + sim * normalized_weight

    if result:
        _log.info(
            "deep_taste multi-embedding: %d candidates across %d prefix types",
            len(result), len(prefixes_and_weights),
        )
    return result, per_type


def _compute_seed_multi_embedding(
    *,
    seed_ids: List[int],
    exclude_scene_ids: Set[int],
    dinov3_weight: float,
    metaclip2_weight: float,
    audio_weight: float,
) -> Dict[int, float]:
    """Seed-based multi-embedding similarity."""
    result: Dict[int, float] = {}
    total_weight = dinov3_weight + metaclip2_weight + audio_weight
    if total_weight <= 0:
        return {}

    prefixes_and_weights = []
    if dinov3_weight > 0:
        prefixes_and_weights.append((VISUAL_DINOV3_PREFIX, dinov3_weight))
    if metaclip2_weight > 0:
        prefixes_and_weights.append((VISUAL_PREFIX, metaclip2_weight))
    if audio_weight > 0:
        for ap in AUDIO_PREFIXES:
            prefixes_and_weights.append((ap, audio_weight / len(AUDIO_PREFIXES)))

    for prefix, weight in prefixes_and_weights:
        try:
            sim_map = find_similar_by_seed_embeddings(
                seed_scene_ids=seed_ids,
                type_prefix=prefix,
                exclude_scene_ids=exclude_scene_ids,
            )
        except Exception:
            _log.debug("deep_taste seed: embedding search failed for %s", prefix, exc_info=True)
            continue

        if sim_map:
            normalized_weight = weight / total_weight
            for sid, sim in sim_map.items():
                if sid in exclude_scene_ids:
                    continue
                result[sid] = result.get(sid, 0.0) + sim * normalized_weight

    return result


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

@recommender(
    id="deep_taste",
    label="Deep Taste",
    description=(
        "Advanced multi-signal recommender with separate visual-style (DINOv3) "
        "and visual-semantic (MetaCLIP2) embedding weights, face-based performer "
        "discovery, cluster-aware diversification, and AI tag temporal matching. "
        "Provides fine-grained control over each signal."
    ),
    contexts=[RecContext.global_feed, RecContext.similar_scene],
    config=[
        # --- Signal weights ---
        {
            "name": "tag_weight",
            "label": "Tag Similarity Weight",
            "type": "slider",
            "default": DEFAULT_TAG_WEIGHT,
            "min": 0.0, "max": 1.5, "step": 0.05,
            "help": "Weight for Stash tag TF-IDF cosine similarity.",
        },
        {
            "name": "performer_weight",
            "label": "Performer Affinity Weight",
            "type": "slider",
            "default": DEFAULT_PERFORMER_WEIGHT,
            "min": 0.0, "max": 1.5, "step": 0.05,
            "help": "Weight for performer co-occurrence affinity.",
        },
        {
            "name": "studio_weight",
            "label": "Studio Affinity Weight",
            "type": "slider",
            "default": DEFAULT_STUDIO_WEIGHT,
            "min": 0.0, "max": 0.5, "step": 0.05,
        },
        {
            "name": "dinov3_weight",
            "label": "Visual Style Weight (DINOv3)",
            "type": "slider",
            "default": DEFAULT_DINOV3_WEIGHT,
            "min": 0.0, "max": 1.5, "step": 0.05,
            "help": "DINOv3 embedding similarity — visual style and appearance matching.",
        },
        {
            "name": "metaclip2_weight",
            "label": "Visual Semantic Weight (MetaCLIP2)",
            "type": "slider",
            "default": DEFAULT_METACLIP2_WEIGHT,
            "min": 0.0, "max": 1.5, "step": 0.05,
            "help": "MetaCLIP2 embedding similarity — semantic scene content matching.",
        },
        {
            "name": "audio_weight",
            "label": "Audio Similarity Weight",
            "type": "slider",
            "default": DEFAULT_AUDIO_WEIGHT,
            "min": 0.0, "max": 1.0, "step": 0.05,
            "help": "Audio embedding similarity (speech, moaning, breathing).",
        },
        {
            "name": "ai_tag_weight",
            "label": "AI Tag Duration Weight",
            "type": "slider",
            "default": DEFAULT_AI_TAG_WEIGHT,
            "min": 0.0, "max": 1.5, "step": 0.05,
            "help": "Duration-weighted AI tag temporal similarity.",
        },
        {
            "name": "face_weight",
            "label": "Face Similarity Weight",
            "type": "slider",
            "default": DEFAULT_FACE_WEIGHT,
            "min": 0.0, "max": 1.0, "step": 0.05,
            "help": "Find scenes with similar faces to liked performers.",
        },
        {
            "name": "negative_weight",
            "label": "Negative Penalty Weight",
            "type": "slider",
            "default": DEFAULT_NEGATIVE_WEIGHT,
            "min": 0.0, "max": 1.0, "step": 0.05,
        },
        # --- Diversity & limits ---
        {
            "name": "cluster_diversity",
            "label": "Cluster Diversity",
            "type": "slider",
            "default": DEFAULT_CLUSTER_DIVERSITY,
            "min": 0.0, "max": 1.0, "step": 0.05,
            "help": "0=pure score ranking, higher=more diverse across content clusters.",
        },
        {
            "name": "target_cluster",
            "label": "Drill Into Cluster",
            "type": "number",
            "default": None,
            "min": -1, "max": 999,
            "help": "Set to a cluster ID to only show results from that cluster (-1 = off).",
        },
        {
            "name": "recency_half_life",
            "label": "Recency Half-Life (days)",
            "type": "slider",
            "default": DEFAULT_RECENCY_HALF_LIFE,
            "min": 0, "max": 180, "step": 5,
        },
        {
            "name": "history_limit",
            "label": "History Limit",
            "type": "number",
            "default": DEFAULT_HISTORY_LIMIT,
            "min": 25, "max": 2000,
        },
        {
            "name": "min_watch_seconds",
            "label": "Min Watch Seconds",
            "type": "number",
            "default": DEFAULT_MIN_WATCH_SECONDS,
            "min": 0, "max": 600,
        },
        {
            "name": "candidate_limit",
            "label": "Max Candidates",
            "type": "number",
            "default": DEFAULT_CANDIDATE_LIMIT,
            "min": 50, "max": 2000,
        },
        # --- Filters ---
        {
            "name": "tag_filter",
            "label": "Tag Filter",
            "type": "tags",
            "default": None,
            "persist": True,
            "help": "Include or exclude scenes by tag. Include = only show scenes with these tags. Exclude = hide scenes with these tags.",
            "constraint_types": ["presence"],
            "allowed_combination_modes": ["or"],
        },
    ],
    supports_pagination=True,
    exposes_scores=True,
    needs_seed_scenes=False,
    allows_multi_seed=True,
)
async def deep_taste(ctx: Dict[str, Any], request: RecommendationRequest):
    """Deep Taste recommender with multi-embedding + face + cluster diversity."""

    cfg = request.config or {}

    # Parse weights
    tag_weight = _coerce_float(cfg.get("tag_weight"), DEFAULT_TAG_WEIGHT)
    performer_weight = _coerce_float(cfg.get("performer_weight"), DEFAULT_PERFORMER_WEIGHT)
    studio_weight = _coerce_float(cfg.get("studio_weight"), DEFAULT_STUDIO_WEIGHT)
    dinov3_weight = _coerce_float(cfg.get("dinov3_weight"), DEFAULT_DINOV3_WEIGHT)
    metaclip2_weight = _coerce_float(cfg.get("metaclip2_weight"), DEFAULT_METACLIP2_WEIGHT)
    audio_weight = _coerce_float(cfg.get("audio_weight"), DEFAULT_AUDIO_WEIGHT)
    ai_tag_weight = _coerce_float(cfg.get("ai_tag_weight"), DEFAULT_AI_TAG_WEIGHT)
    face_weight = _coerce_float(cfg.get("face_weight"), DEFAULT_FACE_WEIGHT)
    negative_weight = _coerce_float(cfg.get("negative_weight"), DEFAULT_NEGATIVE_WEIGHT)
    cluster_diversity = _coerce_float(cfg.get("cluster_diversity"), DEFAULT_CLUSTER_DIVERSITY)
    recency_half_life = _coerce_float(cfg.get("recency_half_life"), DEFAULT_RECENCY_HALF_LIFE)
    history_limit = _coerce_int(cfg.get("history_limit"), DEFAULT_HISTORY_LIMIT)
    min_watch_seconds = _coerce_float(cfg.get("min_watch_seconds"), DEFAULT_MIN_WATCH_SECONDS)
    candidate_limit = _coerce_int(cfg.get("candidate_limit"), DEFAULT_CANDIDATE_LIMIT)
    target_cluster_raw = cfg.get("target_cluster")
    target_cluster: int | None = None
    if target_cluster_raw is not None:
        tc = _coerce_int(target_cluster_raw, -1)
        if tc >= 0:
            target_cluster = tc

    # Combined embedding weight for the generic scoring function
    embedding_weight = dinov3_weight + metaclip2_weight + audio_weight

    # Parse tag filter
    _tag_filter_cfg = cfg.get("tag_filter") or {}
    cfg_include_tag_ids: Set[int] = set()
    cfg_exclude_tag_ids: Set[int] = set()
    if isinstance(_tag_filter_cfg, dict):
        cfg_include_tag_ids = set(_tag_filter_cfg.get("include", []))
        cfg_exclude_tag_ids = set(_tag_filter_cfg.get("exclude", []))

    # Route to handler
    seed_ids = [int(s) for s in request.seedSceneIds or [] if s is not None]
    is_similar_scene = request.context == RecContext.similar_scene and seed_ids

    if is_similar_scene:
        return await _handle_similar(
            seed_ids=seed_ids,
            tag_weight=tag_weight,
            performer_weight=performer_weight,
            studio_weight=studio_weight,
            dinov3_weight=dinov3_weight,
            metaclip2_weight=metaclip2_weight,
            audio_weight=audio_weight,
            embedding_weight=embedding_weight,
            ai_tag_weight=ai_tag_weight,
            negative_weight=negative_weight,
            candidate_limit=candidate_limit,
            cfg_include_tag_ids=cfg_include_tag_ids,
            cfg_exclude_tag_ids=cfg_exclude_tag_ids,
            request=request,
        )

    return await _handle_global(
        tag_weight=tag_weight,
        performer_weight=performer_weight,
        studio_weight=studio_weight,
        dinov3_weight=dinov3_weight,
        metaclip2_weight=metaclip2_weight,
        audio_weight=audio_weight,
        embedding_weight=embedding_weight,
        ai_tag_weight=ai_tag_weight,
        face_weight=face_weight,
        negative_weight=negative_weight,
        cluster_diversity=cluster_diversity,
        target_cluster=target_cluster,
        recency_half_life=recency_half_life,
        history_limit=history_limit,
        min_watch_seconds=min_watch_seconds,
        candidate_limit=candidate_limit,
        cfg_include_tag_ids=cfg_include_tag_ids,
        cfg_exclude_tag_ids=cfg_exclude_tag_ids,
        request=request,
    )


# ---------------------------------------------------------------------------
# Global feed handler
# ---------------------------------------------------------------------------

async def _handle_global(
    *,
    tag_weight: float,
    performer_weight: float,
    studio_weight: float,
    dinov3_weight: float,
    metaclip2_weight: float,
    audio_weight: float,
    embedding_weight: float,
    ai_tag_weight: float,
    face_weight: float,
    negative_weight: float,
    cluster_diversity: float,
    target_cluster: int | None,
    recency_half_life: float,
    history_limit: int,
    min_watch_seconds: float,
    candidate_limit: int,
    cfg_include_tag_ids: Set[int],
    cfg_exclude_tag_ids: Set[int],
    request: RecommendationRequest,
) -> Dict[str, Any]:
    _empty = {"scenes": [], "total": 0, "has_more": False}

    # ── Step 1: Load watch history and rated scenes ──
    history = load_watch_history_summary(
        min_watch_seconds=min_watch_seconds, limit=history_limit,
    )
    watched_scene_ids = {e["scene_id"] for e in history} if history else set()

    from stash_ai_server.services.taste_compute import (
        _get_rated_only_scene_ids,
        _get_rating_scores_for_scenes,
    )
    rated_only_ids = _get_rated_only_scene_ids(watched_scene_ids)
    if rated_only_ids:
        watched_scene_ids = watched_scene_ids | rated_only_ids

    if not watched_scene_ids:
        return _empty

    # ── Step 2: Engagement scores ──
    engagement_map: Dict[int, float] = {}
    try:
        eng_results = score_all_watched_scenes(limit=history_limit, include_rated=True)
        engagement_map = {r.entity_id: r.score for r in eng_results}
    except Exception:
        _log.debug("deep_taste: engagement scores unavailable")

    if rated_only_ids:
        rating_scores = _get_rating_scores_for_scenes(rated_only_ids)
        for sid in rated_only_ids:
            if sid not in engagement_map:
                engagement_map[sid] = rating_scores.get(sid, 0.5)

    # ── Step 3: Corpus / IDF / tag profiles ──
    blacklisted = resolve_blacklisted_tag_ids()
    corpus = fetch_all_scene_tag_ids(exclude_tag_ids=blacklisted)
    if not corpus:
        return _empty
    df_map, total_docs = compute_document_frequencies(corpus)
    idf = compute_idf(df_map, total_docs)

    _log.info("deep_taste: corpus %d scenes, %d unique tags", total_docs, len(df_map))

    watched_tags = fetch_scene_tag_ids(list(watched_scene_ids), exclude_tag_ids=blacklisted)
    watched_payloads = fetch_scenes_by_ids(list(watched_scene_ids))
    watched_performers = _extract_scene_performers(watched_payloads)
    watched_studios = _extract_scene_studio_ids(watched_payloads)

    # Taste weights with recency + data-depth
    taste_weights = engagement_map
    if engagement_map and recency_half_life > 0:
        depth_scores = compute_data_depth(
            list(watched_scene_ids), corpus_tags=corpus, performer_map=watched_performers,
        )
        taste_weights = build_taste_weights(
            engagement_map, history,
            half_life_days=recency_half_life, data_depth=depth_scores,
        )

    user_profile = build_user_tag_profile(
        watched_scene_tags=watched_tags, idf=idf,
        engagement_scores=taste_weights if taste_weights else None,
    )
    performer_profile = build_user_performer_profile(
        watched_scene_performers=watched_performers,
        engagement_scores=taste_weights if taste_weights else None,
    )
    studio_affinity = _build_studio_affinity(watched_studios, taste_weights if taste_weights else None)

    # ── Step 4: Negative profiles ──
    negative_tag_profile: Dict[int, float] = {}
    negative_performer_prof: Dict[int, float] = {}
    if negative_weight > 0 and engagement_map:
        liked_ids, disliked_ids = split_by_engagement(list(watched_scene_ids), engagement_map)
        if disliked_ids:
            disliked_tags = {s: watched_tags.get(s, set()) for s in disliked_ids if s in watched_tags}
            liked_tags = {s: watched_tags.get(s, set()) for s in liked_ids if s in watched_tags}
            negative_tag_profile = build_negative_tag_profile(
                disliked_scene_tags=disliked_tags, liked_scene_tags=liked_tags, idf=idf,
            )
            disliked_perfs = {s: watched_performers.get(s, set()) for s in disliked_ids if s in watched_performers}
            liked_perfs = {s: watched_performers.get(s, set()) for s in liked_ids if s in watched_performers}
            negative_performer_prof = build_negative_performer_profile(
                disliked_scene_performers=disliked_perfs, liked_scene_performers=liked_perfs,
            )

    # ── Step 5: Candidates (everything not watched) ──
    candidate_ids = [sid for sid in corpus if sid not in watched_scene_ids]
    if not candidate_ids:
        return _empty

    candidate_tags = {sid: corpus[sid] for sid in candidate_ids if sid in corpus}
    candidate_vectors = build_tfidf_vectors(candidate_tags, idf)

    # ── Step 6: Multi-embedding similarity ──
    embedding_sim_map: Dict[int, float] = {}
    embedding_per_type: Dict[str, Dict[int, float]] = {}
    if embedding_weight > 0:
        embedding_sim_map, embedding_per_type = _compute_multi_embedding_similarity(
            watched_scene_ids=watched_scene_ids,
            engagement_scores=taste_weights if taste_weights else None,
            exclude_scene_ids=watched_scene_ids,
            dinov3_weight=dinov3_weight,
            metaclip2_weight=metaclip2_weight,
            audio_weight=audio_weight,
        )

    # ── Step 7: AI tag similarity ──
    ai_tag_sim_map: Dict[int, float] = {}
    ai_tag_detail_map: Dict[int, List] = {}
    if ai_tag_weight > 0:
        try:
            ai_result = compute_ai_tag_similarity(
                watched_scene_ids=list(watched_scene_ids),
                engagement_scores=taste_weights if taste_weights else None,
                candidate_scene_ids=candidate_ids,
                mode="taste",
                return_detail=True,
            )
            ai_tag_sim_map, ai_tag_detail_map = ai_result
        except Exception:
            _log.debug("deep_taste: AI tag similarity unavailable", exc_info=True)

    # ── Step 8: Face similarity ──
    face_sim_map: Dict[int, float] = {}
    if face_weight > 0:
        face_sim_map = _find_scenes_by_liked_faces(
            watched_scene_ids=watched_scene_ids,
            engagement_scores=taste_weights if taste_weights else None,
        )

    # ── Step 9: Pre-filter scoring ──
    def _neg_penalty_tag_only(vec: Mapping) -> float:
        if not negative_tag_profile:
            return 0.0
        from stash_ai_server.recommendations.utils.negative_signals import compute_negative_tag_penalty
        return compute_negative_tag_penalty(vec, negative_tag_profile)

    # Combine face score into embedding score for the generic scorer
    def _combined_embed_score(scene_id: int) -> float:
        e = embedding_sim_map.get(scene_id, 0.0)
        f = face_sim_map.get(scene_id, 0.0)
        if embedding_weight + face_weight <= 0:
            return 0.0
        return (e * embedding_weight + f * face_weight) / (embedding_weight + face_weight)

    scored: List[Tuple[int, float, Dict[str, Any]]] = []
    tag_scored_ids: Set[int] = set()

    for scene_id in candidate_ids:
        vec = candidate_vectors.get(scene_id, {})
        combined_emb = _combined_embed_score(scene_id)

        score, debug = score_scene_against_profile(
            scene_vector=vec,
            user_profile=user_profile,
            scene_performers=None,
            performer_profile=performer_profile,
            scene_studio_id=None,
            studio_affinity=studio_affinity,
            tag_weight=tag_weight,
            performer_weight=performer_weight,
            studio_weight=studio_weight,
            embedding_score=combined_emb,
            embedding_weight=embedding_weight + face_weight,
            ai_tag_score=ai_tag_sim_map.get(scene_id, 0.0),
            ai_tag_weight=ai_tag_weight,
            negative_penalty=_neg_penalty_tag_only(vec),
            negative_weight=negative_weight,
        )
        if score > 0:
            scored.append((scene_id, score, debug))
            tag_scored_ids.add(scene_id)

    # Add embedding/face/AI-tag-only candidates not found by pre-filter
    extra_sources = {}
    for sid, sim in embedding_sim_map.items():
        if sid not in tag_scored_ids and sid not in watched_scene_ids and sim > 0:
            extra_sources[sid] = max(extra_sources.get(sid, 0.0), sim)
    for sid, sim in face_sim_map.items():
        if sid not in tag_scored_ids and sid not in watched_scene_ids and sim > 0:
            extra_sources[sid] = max(extra_sources.get(sid, 0.0), sim)
    for sid, sim in ai_tag_sim_map.items():
        if sid not in tag_scored_ids and sid not in watched_scene_ids and sim > 0:
            extra_sources[sid] = max(extra_sources.get(sid, 0.0), sim)

    for scene_id in extra_sources:
        combined_emb = _combined_embed_score(scene_id)
        score, debug = score_scene_against_profile(
            scene_vector={},
            user_profile=user_profile,
            scene_performers=None,
            performer_profile=performer_profile,
            scene_studio_id=None,
            studio_affinity=studio_affinity,
            tag_weight=tag_weight,
            performer_weight=performer_weight,
            studio_weight=studio_weight,
            embedding_score=combined_emb,
            embedding_weight=embedding_weight + face_weight,
            ai_tag_score=ai_tag_sim_map.get(scene_id, 0.0),
            ai_tag_weight=ai_tag_weight,
            negative_penalty=0.0,
            negative_weight=negative_weight,
        )
        if score > 0:
            scored.append((scene_id, score, debug))
            tag_scored_ids.add(scene_id)
            if scene_id not in candidate_tags:
                candidate_tags[scene_id] = corpus.get(scene_id, set())

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:candidate_limit]

    if not scored:
        return _empty

    # ── Step 10: Full re-scoring with performer/studio + negative ──
    top_ids = [s[0] for s in scored]
    candidate_payloads = fetch_scenes_by_ids(top_ids)
    candidate_performer_map = _extract_scene_performers(candidate_payloads)
    candidate_studio_map = _extract_scene_studio_ids(candidate_payloads)

    fully_scored: List[Tuple[int, float, Dict[str, Any]]] = []
    for scene_id, _, _ in scored:
        vec = candidate_vectors.get(scene_id, {})
        neg_pen = compute_combined_negative_penalty(
            scene_vector=vec,
            negative_tag_profile=negative_tag_profile,
            scene_performers=candidate_performer_map.get(scene_id),
            negative_performer_profile=negative_performer_prof if negative_performer_prof else None,
        ) if negative_weight > 0 else 0.0

        combined_emb = _combined_embed_score(scene_id)

        score, debug = score_scene_against_profile(
            scene_vector=vec,
            user_profile=user_profile,
            scene_performers=candidate_performer_map.get(scene_id),
            performer_profile=performer_profile,
            scene_studio_id=candidate_studio_map.get(scene_id),
            studio_affinity=studio_affinity,
            tag_weight=tag_weight,
            performer_weight=performer_weight,
            studio_weight=studio_weight,
            embedding_score=combined_emb,
            embedding_weight=embedding_weight + face_weight,
            ai_tag_score=ai_tag_sim_map.get(scene_id, 0.0),
            ai_tag_weight=ai_tag_weight,
            negative_penalty=neg_pen,
            negative_weight=negative_weight,
        )
        fully_scored.append((scene_id, score, debug))

    fully_scored.sort(key=lambda x: x[1], reverse=True)

    # ── Step 11: Tag filtering (include + exclude) ──
    if cfg_exclude_tag_ids:
        before = len(fully_scored)
        fully_scored = [
            (sid, sc, dbg) for sid, sc, dbg in fully_scored
            if not (candidate_tags.get(sid, set()) & cfg_exclude_tag_ids)
        ]
        _log.info("deep_taste: excluded %d scenes via tag_filter exclude", before - len(fully_scored))

    if cfg_include_tag_ids:
        before = len(fully_scored)
        fully_scored = [
            (sid, sc, dbg) for sid, sc, dbg in fully_scored
            if (candidate_tags.get(sid, set()) & cfg_include_tag_ids)
        ]
        _log.info("deep_taste: include filter narrowed from %d to %d scenes", before, len(fully_scored))

    # ── Step 12: Cluster-aware diversification ──
    scene_clusters = _load_scene_clusters() if cluster_diversity > 0 or target_cluster is not None else {}
    cluster_affinities = _load_cluster_affinities() if scene_clusters else {}

    if not scene_clusters and (cluster_diversity > 0 or target_cluster is not None):
        _log.warning(
            "deep_taste: no content clusters found — run 'Compute Clusters' "
            "from the Training page to enable cluster diversity / drill-down"
        )

    if scene_clusters and (cluster_diversity > 0 or target_cluster is not None):
        fully_scored = _diversify_by_cluster(
            fully_scored, scene_clusters, cluster_affinities,
            diversity_factor=cluster_diversity,
            target_cluster_id=target_cluster,
        )

    total = len(fully_scored)

    # ── Step 13: Tag names for debug ──
    all_tag_ids: Set[int] = set()
    for scene_id, _, _ in fully_scored:
        all_tag_ids.update(candidate_tags.get(scene_id, set()))
    for tid in user_profile:
        all_tag_ids.add(tid)
    for tid in negative_tag_profile:
        all_tag_ids.add(tid)
    # Also include AI tag IDs from detail maps
    for sid_list in ai_tag_detail_map.values():
        for entry in sid_list:
            all_tag_ids.add(entry["tag_id"])
    tag_names = fetch_tag_names(all_tag_ids)

    # ── Step 14: Paginate + build results ──
    offset = max(0, request.offset or 0)
    limit = max(1, request.limit or 40)
    page = fully_scored[offset:offset + limit]

    if not page:
        return {"scenes": [], "total": total, "has_more": False}

    results: List[Dict[str, Any]] = []
    for scene_id, score, debug in page:
        payload = candidate_payloads.get(scene_id)
        if payload is None:
            continue

        tags = candidate_tags.get(scene_id, set())
        top_tags = _annotate_top_tags(tags, user_profile, idf, tag_names)
        vec = candidate_vectors.get(scene_id, {})
        tag_contribs = _compute_tag_contributions(vec, user_profile, tag_names)
        perf_detail = _build_performer_detail(
            candidate_performer_map.get(scene_id),
            performer_profile,
            candidate_payloads,
        )

        # AI tag detail with resolved names
        ai_detail = ai_tag_detail_map.get(scene_id, [])
        for entry in ai_detail:
            entry["tag_name"] = tag_names.get(entry["tag_id"], f"ai_tag_{entry['tag_id']}")

        # Negative detail
        neg_detail: Dict[str, Any] = {}
        if negative_weight > 0 and (negative_tag_profile or negative_performer_prof):
            neg_detail = compute_negative_detail(
                scene_vector=vec,
                negative_tag_profile=negative_tag_profile,
                scene_performers=candidate_performer_map.get(scene_id),
                negative_performer_profile=negative_performer_prof if negative_performer_prof else None,
            )
            for entry in neg_detail.get("tag_contributions", []):
                entry["tag_name"] = tag_names.get(entry["tag_id"], f"tag_{entry['tag_id']}")

        cluster_id = scene_clusters.get(scene_id)

        payload["score"] = round(score, 4)
        payload["debug_meta"] = {
            "deep_taste": {
                "score": round(score, 4),
                "tag_similarity": debug["tag_similarity"],
                "performer_score": debug["performer_score"],
                "studio_score": debug["studio_score"],
                "embedding_score": debug["embedding_score"],
                "ai_tag_score": debug["ai_tag_score"],
                "face_sim": round(face_sim_map.get(scene_id, 0.0), 4),
                "dinov3_sim": round(embedding_per_type.get("dinov3", {}).get(scene_id, 0.0), 4),
                "metaclip2_sim": round(embedding_per_type.get("metaclip2", {}).get(scene_id, 0.0), 4),
                "audio_sim": round(embedding_per_type.get("audio", {}).get(scene_id, 0.0), 4),
                "negative_penalty": debug.get("negative_penalty", 0.0),
                "matched_performers": debug["matched_performers"],
                "cluster_id": cluster_id,
                "clusters_available": bool(scene_clusters),
                "weights": {
                    "tag": tag_weight,
                    "performer": performer_weight,
                    "studio": studio_weight,
                    "dinov3": dinov3_weight,
                    "metaclip2": metaclip2_weight,
                    "audio": audio_weight,
                    "ai_tag": ai_tag_weight,
                    "face": face_weight,
                    "negative": negative_weight,
                },
                "top_matching_tags": top_tags,
                "tag_contributions": tag_contribs,
                "performer_detail": perf_detail,
                "ai_tag_detail": ai_detail,
                "negative_detail": neg_detail,
                "studio_name": None,
                "profile_tags": len(user_profile),
                "negative_tags": len(negative_tag_profile),
                "recency_half_life": recency_half_life,
                "scene_tags": len(tags),
                "corpus_size": total + len(watched_scene_ids),
            },
        }

        # Resolve studio name
        s_id = candidate_studio_map.get(scene_id)
        if s_id is not None:
            s_payload = candidate_payloads.get(scene_id, {})
            s_info = s_payload.get("studio")
            if isinstance(s_info, dict):
                payload["debug_meta"]["deep_taste"]["studio_name"] = s_info.get("name")

        results.append(payload)

    has_more = offset + len(page) < total
    return {"scenes": results, "total": total, "has_more": has_more}


# ---------------------------------------------------------------------------
# Similar scene handler
# ---------------------------------------------------------------------------

async def _handle_similar(
    *,
    seed_ids: List[int],
    tag_weight: float,
    performer_weight: float,
    studio_weight: float,
    dinov3_weight: float,
    metaclip2_weight: float,
    audio_weight: float,
    embedding_weight: float,
    ai_tag_weight: float,
    negative_weight: float,
    candidate_limit: int,
    cfg_include_tag_ids: Set[int],
    cfg_exclude_tag_ids: Set[int],
    request: RecommendationRequest,
) -> Dict[str, Any]:
    _empty = {"scenes": [], "total": 0, "has_more": False}

    blacklisted = resolve_blacklisted_tag_ids()
    corpus = fetch_all_scene_tag_ids(exclude_tag_ids=blacklisted)
    if not corpus:
        return _empty
    df_map, total_docs = compute_document_frequencies(corpus)
    idf = compute_idf(df_map, total_docs)

    seed_tags = fetch_scene_tag_ids(seed_ids, exclude_tag_ids=blacklisted)
    seed_profile = build_user_tag_profile(
        watched_scene_tags=seed_tags, idf=idf, engagement_scores=None,
    ) if seed_tags else {}

    if not seed_profile and embedding_weight <= 0 and ai_tag_weight <= 0:
        return _empty

    seed_payloads = fetch_scenes_by_ids(seed_ids)
    seed_performers = _extract_scene_performers(seed_payloads)
    seed_studios = _extract_scene_studio_ids(seed_payloads)

    all_seed_performers: Set[int] = set()
    for pids in seed_performers.values():
        all_seed_performers.update(pids)
    performer_profile = {pid: 1.0 for pid in all_seed_performers}
    studio_affinity = {sid: 1.0 for sid in set(seed_studios.values())}

    seed_set = set(seed_ids)

    # Multi-embedding similarity for seed scenes
    embedding_sim_map: Dict[int, float] = {}
    if embedding_weight > 0:
        embedding_sim_map = _compute_seed_multi_embedding(
            seed_ids=seed_ids,
            exclude_scene_ids=seed_set,
            dinov3_weight=dinov3_weight,
            metaclip2_weight=metaclip2_weight,
            audio_weight=audio_weight,
        )

    # AI tag similarity
    ai_tag_sim_map: Dict[int, float] = {}
    if ai_tag_weight > 0:
        try:
            candidate_ids_for_ai = [sid for sid in corpus if sid not in seed_set]
            ai_tag_sim_map = compute_ai_tag_similarity(
                watched_scene_ids=seed_ids,
                engagement_scores=None,
                candidate_scene_ids=candidate_ids_for_ai,
                mode="seed",
            )
        except Exception:
            _log.debug("deep_taste similar: AI tag unavailable", exc_info=True)

    # Score candidates
    candidate_ids = [sid for sid in corpus if sid not in seed_set]
    candidate_tags = {sid: corpus[sid] for sid in candidate_ids if sid in corpus}
    candidate_vectors = build_tfidf_vectors(candidate_tags, idf)

    scored: List[Tuple[int, float, Dict[str, Any]]] = []
    tag_scored_ids: Set[int] = set()

    for scene_id in candidate_ids:
        vec = candidate_vectors.get(scene_id, {})
        score, debug = score_scene_against_profile(
            scene_vector=vec,
            user_profile=seed_profile,
            scene_performers=None,
            performer_profile=performer_profile,
            scene_studio_id=None,
            studio_affinity=studio_affinity,
            tag_weight=tag_weight,
            performer_weight=performer_weight,
            studio_weight=studio_weight,
            embedding_score=embedding_sim_map.get(scene_id, 0.0),
            embedding_weight=embedding_weight,
            ai_tag_score=ai_tag_sim_map.get(scene_id, 0.0),
            ai_tag_weight=ai_tag_weight,
        )
        if score > 0:
            scored.append((scene_id, score, debug))
            tag_scored_ids.add(scene_id)

    # Add embedding/AI-tag-only candidates
    for sid, sim in {**embedding_sim_map, **ai_tag_sim_map}.items():
        if sid in tag_scored_ids or sid in seed_set or sim <= 0:
            continue
        score, debug = score_scene_against_profile(
            scene_vector={},
            user_profile=seed_profile,
            scene_performers=None,
            performer_profile=performer_profile,
            scene_studio_id=None,
            studio_affinity=studio_affinity,
            tag_weight=tag_weight,
            performer_weight=performer_weight,
            studio_weight=studio_weight,
            embedding_score=embedding_sim_map.get(sid, 0.0),
            embedding_weight=embedding_weight,
            ai_tag_score=ai_tag_sim_map.get(sid, 0.0),
            ai_tag_weight=ai_tag_weight,
        )
        if score > 0:
            scored.append((sid, score, debug))
            tag_scored_ids.add(sid)
            if sid not in candidate_tags:
                candidate_tags[sid] = corpus.get(sid, set())

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:candidate_limit]

    if not scored:
        return _empty

    # Hydrate + re-score with performer/studio
    top_ids = [s[0] for s in scored]
    candidate_payloads = fetch_scenes_by_ids(top_ids)
    candidate_performer_map = _extract_scene_performers(candidate_payloads)
    candidate_studio_map = _extract_scene_studio_ids(candidate_payloads)

    fully_scored: List[Tuple[int, float, Dict[str, Any]]] = []
    for scene_id, _, _ in scored:
        vec = candidate_vectors.get(scene_id, {})
        score, debug = score_scene_against_profile(
            scene_vector=vec,
            user_profile=seed_profile,
            scene_performers=candidate_performer_map.get(scene_id),
            performer_profile=performer_profile,
            scene_studio_id=candidate_studio_map.get(scene_id),
            studio_affinity=studio_affinity,
            tag_weight=tag_weight,
            performer_weight=performer_weight,
            studio_weight=studio_weight,
            embedding_score=embedding_sim_map.get(scene_id, 0.0),
            embedding_weight=embedding_weight,
            ai_tag_score=ai_tag_sim_map.get(scene_id, 0.0),
            ai_tag_weight=ai_tag_weight,
        )
        fully_scored.append((scene_id, score, debug))

    fully_scored.sort(key=lambda x: x[1], reverse=True)

    # Tag filtering (include + exclude)
    if cfg_exclude_tag_ids:
        fully_scored = [
            (sid, sc, dbg) for sid, sc, dbg in fully_scored
            if not (candidate_tags.get(sid, set()) & cfg_exclude_tag_ids)
        ]
    if cfg_include_tag_ids:
        fully_scored = [
            (sid, sc, dbg) for sid, sc, dbg in fully_scored
            if (candidate_tags.get(sid, set()) & cfg_include_tag_ids)
        ]

    total = len(fully_scored)

    # Tag names
    all_tag_ids: Set[int] = set()
    for scene_id, _, _ in fully_scored:
        all_tag_ids.update(candidate_tags.get(scene_id, set()))
    for tid in seed_profile:
        all_tag_ids.add(tid)
    tag_names = fetch_tag_names(all_tag_ids)

    # Paginate
    offset = max(0, request.offset or 0)
    limit = max(1, request.limit or 40)
    page = fully_scored[offset:offset + limit]

    if not page:
        return {"scenes": [], "total": total, "has_more": False}

    results: List[Dict[str, Any]] = []
    for scene_id, score, debug in page:
        payload = candidate_payloads.get(scene_id)
        if payload is None:
            continue

        tags = candidate_tags.get(scene_id, set())
        top_tags_list = _annotate_top_tags(tags, seed_profile, idf, tag_names)

        payload["score"] = round(score, 4)
        payload["debug_meta"] = {
            "deep_taste": {
                "score": round(score, 4),
                "tag_similarity": debug["tag_similarity"],
                "performer_score": debug["performer_score"],
                "studio_score": debug["studio_score"],
                "embedding_score": debug["embedding_score"],
                "ai_tag_score": debug["ai_tag_score"],
                "matched_performers": debug["matched_performers"],
                "weights": {
                    "tag": tag_weight,
                    "performer": performer_weight,
                    "studio": studio_weight,
                    "dinov3": dinov3_weight,
                    "metaclip2": metaclip2_weight,
                    "audio": audio_weight,
                    "ai_tag": ai_tag_weight,
                    "negative": negative_weight,
                },
                "top_matching_tags": top_tags_list,
                "seed_scene_ids": seed_ids,
                "scene_tags": len(tags),
                "corpus_size": len(corpus),
            },
        }
        results.append(payload)

    has_more = offset + len(page) < total
    return {"scenes": results, "total": total, "has_more": has_more}
