"""Content-based recommendation plugin using Stash scene-level tags.

Works with the tags every Stash user already has — no AI tagging required.
Supports both ``global_feed`` (discover new content matching your taste) and
``similar_scene`` (find scenes like a given seed) contexts.

Scoring blends five positive signals and one negative signal:
  1. **Tag similarity**  – TF-IDF cosine similarity between the candidate scene
     and the user's taste profile (or seed scene for similar_scene context).
  2. **Performer affinity** – How often the user has watched this performer,
     weighted by engagement score.
  3. **Studio affinity** – Minor bonus for studios the user tends to engage with.
  4. **Embedding similarity** – MetaCLIP2 visual + ECAPA-TDNN audio embeddings.
  5. **AI tag duration similarity** – Temporal tag data from AI tagging.
  6. **Negative penalty** – Penalises candidates matching patterns from content
     the user watched but didn't enjoy (low engagement).

Additional taste-profile enhancements:
  - **Recency boost** – Recent watches influence preferences more than old ones
    via exponential time-decay (configurable half-life).
  - **Data-depth weighting** – Reference scenes with richer data (more tags,
    embeddings, AI tags) are weighted higher when building taste centroids.

Each signal contributes to a [0, 1] composite score with configurable weights.
Full debug metadata is attached so the user can inspect why each scene was
recommended and verify the system is working.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Set, Tuple

from stash_ai_server.recommendations.engagement.scorer import (
    score_all_watched_scenes,
)
from stash_ai_server.recommendations.models import RecContext, RecommendationRequest
from stash_ai_server.recommendations.registry import recommender
from stash_ai_server.recommendations.utils.scene_fetch import fetch_scenes_by_ids
from stash_ai_server.recommendations.utils.stash_tags import (
    build_tfidf_vector,
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
    compute_embedding_similarity,
    find_similar_by_cached_centroids,
)
from stash_ai_server.recommendations.utils.ai_tag_similarity import (
    compute_ai_tag_similarity,
)
from stash_ai_server.recommendations.utils.taste_weighting import (
    apply_recency_boost,
    build_taste_weights,
    compute_data_depth,
    DEFAULT_HALF_LIFE_DAYS,
)
from stash_ai_server.recommendations.utils.negative_signals import (
    build_negative_performer_profile,
    build_negative_tag_profile,
    compute_combined_negative_penalty,
    split_by_engagement,
)

_log = logging.getLogger(__name__)

DEFAULT_TAG_WEIGHT = 0.6
DEFAULT_PERFORMER_WEIGHT = 0.3
DEFAULT_STUDIO_WEIGHT = 0.05
DEFAULT_EMBEDDING_WEIGHT = 0.25
DEFAULT_AI_TAG_WEIGHT = 0.35
DEFAULT_NEGATIVE_WEIGHT = 0.15
DEFAULT_RECENCY_HALF_LIFE = DEFAULT_HALF_LIFE_DAYS  # 30 days
DEFAULT_HISTORY_LIMIT = 400
DEFAULT_MIN_WATCH_SECONDS = 15.0
DEFAULT_CANDIDATE_LIMIT = 300
DEFAULT_TOP_TAG_DEBUG = 8


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


# ---------------------------------------------------------------------------
# Helpers to extract performer/studio data from hydrated payloads
# ---------------------------------------------------------------------------

def _extract_scene_performers(payloads: Mapping[int, Dict[str, Any]]) -> Dict[int, Set[int]]:
    """Return ``{scene_id: {performer_id, ...}}`` from hydrated scene payloads."""
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
    """Return ``{scene_id: studio_id}`` from hydrated scene payloads."""
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
    """Build ``{studio_id: affinity}`` normalized to [0, 1]."""
    raw: Dict[int, float] = defaultdict(float)
    for scene_id, studio_id in studio_map.items():
        weight = 1.0
        if engagement_scores:
            weight = max(engagement_scores.get(scene_id, 0.0), 0.0)
            if weight <= 0:
                continue
        raw[studio_id] += weight
    if not raw:
        return {}
    max_val = max(raw.values())
    if max_val <= 0:
        return {}
    return {sid: v / max_val for sid, v in raw.items()}


def _annotate_top_tags(
    scene_tags: Set[int],
    user_profile: Mapping[int, float],
    idf: Mapping[int, float],
    tag_names: Mapping[int, str],
    limit: int = DEFAULT_TOP_TAG_DEBUG,
) -> List[Dict[str, Any]]:
    """Return the top contributing tags for debug display."""
    contributions: List[Tuple[int, float]] = []
    for tid in scene_tags:
        profile_weight = user_profile.get(tid, 0.0)
        idf_val = idf.get(tid, 0.0)
        if profile_weight > 0:
            contributions.append((tid, profile_weight * idf_val))
    contributions.sort(key=lambda x: x[1], reverse=True)
    return [
        {
            "tag_id": tid,
            "tag_name": tag_names.get(tid, f"tag_{tid}"),
            "profile_weight": round(user_profile.get(tid, 0.0), 4),
            "idf": round(idf.get(tid, 0.0), 4),
        }
        for tid, _ in contributions[:limit]
    ]


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

@recommender(
    id="content_based",
    label="Content-Based (Stash Tags)",
    description=(
        "Recommends scenes by matching your Stash tag preferences, performer "
        "history, and studio affinity. Works with tags every user already has — "
        "no AI tagging required. Shows full debug breakdown for tuning."
    ),
    contexts=[RecContext.global_feed, RecContext.similar_scene],
    config=[
        {
            "name": "tag_weight",
            "label": "Tag Weight",
            "type": "slider",
            "default": DEFAULT_TAG_WEIGHT,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "How much tag similarity contributes to the score.",
        },
        {
            "name": "performer_weight",
            "label": "Performer Weight",
            "type": "slider",
            "default": DEFAULT_PERFORMER_WEIGHT,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "How much performer overlap contributes to the score.",
        },
        {
            "name": "studio_weight",
            "label": "Studio Weight",
            "type": "slider",
            "default": DEFAULT_STUDIO_WEIGHT,
            "min": 0.0,
            "max": 0.3,
            "step": 0.05,
            "help": "Minor bonus for studios you tend to watch.",
        },
        {
            "name": "embedding_weight",
            "label": "Embedding Weight",
            "type": "slider",
            "default": DEFAULT_EMBEDDING_WEIGHT,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": (
                "How much visual/audio embedding similarity contributes. "
                "Requires AI tagging to have generated embeddings. Set to 0 to disable."
            ),
        },
        {
            "name": "ai_tag_weight",
            "label": "AI Tag Weight",
            "type": "slider",
            "default": DEFAULT_AI_TAG_WEIGHT,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": (
                "How much AI-detected tag duration similarity contributes. "
                "Uses temporal data (how long each tag appears) for richer matching. "
                "Requires scenes to have been AI-tagged. Set to 0 to disable."
            ),
        },
        {
            "name": "negative_weight",
            "label": "Negative Signal Weight",
            "type": "slider",
            "default": DEFAULT_NEGATIVE_WEIGHT,
            "min": 0.0,
            "max": 0.5,
            "step": 0.05,
            "help": (
                "How much to penalise scenes matching patterns from content you "
                "disliked. Higher = stronger avoidance. Set to 0 to disable."
            ),
        },
        {
            "name": "recency_half_life",
            "label": "Recency Half-Life (days)",
            "type": "slider",
            "default": DEFAULT_RECENCY_HALF_LIFE,
            "min": 0,
            "max": 180,
            "step": 5,
            "help": (
                "Recent watches influence your taste more. This is the number of "
                "days for a watch's influence to halve. 0 = recency disabled."
            ),
        },
        {
            "name": "history_limit",
            "label": "History Scene Limit",
            "type": "number",
            "default": DEFAULT_HISTORY_LIMIT,
            "min": 25,
            "max": 1000,
            "help": "Max number of watched scenes used to build your taste profile.",
        },
        {
            "name": "min_watch_seconds",
            "label": "Min Watch Seconds",
            "type": "number",
            "default": DEFAULT_MIN_WATCH_SECONDS,
            "min": 0,
            "max": 600,
            "help": "Minimum seconds watched for a scene to count in your profile.",
        },
        {
            "name": "candidate_limit",
            "label": "Candidate Pool Size",
            "type": "number",
            "default": DEFAULT_CANDIDATE_LIMIT,
            "min": 50,
            "max": 2000,
            "help": "Max candidates scored before pagination. Larger = slower but more thorough.",
        },
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
async def content_based(ctx: Dict[str, Any], request: RecommendationRequest):
    """Content-based recommendation using Stash tags + performers + studio."""

    cfg = request.config or {}
    tag_weight = _coerce_float(cfg.get("tag_weight"), DEFAULT_TAG_WEIGHT)
    performer_weight = _coerce_float(cfg.get("performer_weight"), DEFAULT_PERFORMER_WEIGHT)
    studio_weight = _coerce_float(cfg.get("studio_weight"), DEFAULT_STUDIO_WEIGHT)
    embedding_weight = _coerce_float(cfg.get("embedding_weight"), DEFAULT_EMBEDDING_WEIGHT)
    ai_tag_weight = _coerce_float(cfg.get("ai_tag_weight"), DEFAULT_AI_TAG_WEIGHT)
    negative_weight = _coerce_float(cfg.get("negative_weight"), DEFAULT_NEGATIVE_WEIGHT)
    recency_half_life = _coerce_float(cfg.get("recency_half_life"), DEFAULT_RECENCY_HALF_LIFE)
    history_limit = _coerce_int(cfg.get("history_limit"), DEFAULT_HISTORY_LIMIT)
    min_watch_seconds = _coerce_float(cfg.get("min_watch_seconds"), DEFAULT_MIN_WATCH_SECONDS)
    candidate_limit = _coerce_int(cfg.get("candidate_limit"), DEFAULT_CANDIDATE_LIMIT)

    # Parse tag filter from plugin config fields
    _tag_filter_cfg = cfg.get("tag_filter") or {}
    cfg_include_tag_ids: Set[int] = set()
    cfg_exclude_tag_ids: Set[int] = set()
    if isinstance(_tag_filter_cfg, dict):
        cfg_include_tag_ids = set(_tag_filter_cfg.get("include", []))
        cfg_exclude_tag_ids = set(_tag_filter_cfg.get("exclude", []))

    # ------------------------------------------------------------------
    # Step 1: Build corpus-level IDF index from Stash tags
    # ------------------------------------------------------------------
    blacklisted_tag_ids = resolve_blacklisted_tag_ids()
    corpus = fetch_all_scene_tag_ids(exclude_tag_ids=blacklisted_tag_ids)
    if not corpus:
        _log.warning("content_based: no Stash tags found in corpus")
        return {"scenes": [], "total": 0, "has_more": False}

    df_map, total_docs = compute_document_frequencies(corpus)
    idf = compute_idf(df_map, total_docs)

    _log.info(
        "content_based: corpus has %d scenes, %d unique tags",
        total_docs, len(df_map),
    )

    # ------------------------------------------------------------------
    # Step 2: Build the reference profile (user taste or seed scenes)
    # ------------------------------------------------------------------
    seed_ids = [int(s) for s in request.seedSceneIds or [] if s is not None]
    is_similar_scene = request.context == RecContext.similar_scene and seed_ids

    if is_similar_scene:
        return await _handle_similar_scene(
            seed_ids=seed_ids,
            corpus=corpus,
            idf=idf,
            df_map=df_map,
            tag_weight=tag_weight,
            performer_weight=performer_weight,
            studio_weight=studio_weight,
            embedding_weight=embedding_weight,
            ai_tag_weight=ai_tag_weight,
            negative_weight=negative_weight,
            candidate_limit=candidate_limit,
            blacklisted_tag_ids=blacklisted_tag_ids,
            cfg_include_tag_ids=cfg_include_tag_ids,
            cfg_exclude_tag_ids=cfg_exclude_tag_ids,
            request=request,
        )

    return await _handle_global_feed(
        corpus=corpus,
        idf=idf,
        df_map=df_map,
        tag_weight=tag_weight,
        performer_weight=performer_weight,
        studio_weight=studio_weight,
        embedding_weight=embedding_weight,
        ai_tag_weight=ai_tag_weight,
        negative_weight=negative_weight,
        recency_half_life=recency_half_life,
        history_limit=history_limit,
        min_watch_seconds=min_watch_seconds,
        candidate_limit=candidate_limit,
        blacklisted_tag_ids=blacklisted_tag_ids,
        cfg_include_tag_ids=cfg_include_tag_ids,
        cfg_exclude_tag_ids=cfg_exclude_tag_ids,
        request=request,
    )


# ---------------------------------------------------------------------------
# global_feed: recommend based on user taste profile
# ---------------------------------------------------------------------------

async def _handle_global_feed(
    *,
    corpus: Dict[int, Set[int]],
    idf: Dict[int, float],
    df_map: Dict[int, int],
    tag_weight: float,
    performer_weight: float,
    studio_weight: float,
    embedding_weight: float,
    ai_tag_weight: float,
    negative_weight: float,
    recency_half_life: float,
    history_limit: int,
    min_watch_seconds: float,
    candidate_limit: int,
    blacklisted_tag_ids: Set[int],
    cfg_include_tag_ids: Set[int],
    cfg_exclude_tag_ids: Set[int],
    request: RecommendationRequest,
) -> Dict[str, Any]:
    # Load watch history
    history = load_watch_history_summary(
        min_watch_seconds=min_watch_seconds,
        limit=history_limit,
    )

    watched_scene_ids = {entry["scene_id"] for entry in history} if history else set()

    # Include rated-but-unwatched scenes in the reference set
    from stash_ai_server.services.taste_compute import (
        _get_rated_only_scene_ids,
        _get_rating_scores_for_scenes,
    )
    rated_only_ids = _get_rated_only_scene_ids(watched_scene_ids)
    if rated_only_ids:
        _log.info(
            "content_based: adding %d rated-but-unwatched scenes to reference set",
            len(rated_only_ids),
        )
        watched_scene_ids = watched_scene_ids | rated_only_ids

    if not watched_scene_ids:
        _log.info("content_based: no watch history or rated scenes found")
        return {"scenes": [], "total": 0, "has_more": False}

    _log.info("content_based: %d reference scenes (watched + rated)", len(watched_scene_ids))

    # Get engagement scores to weight the taste profile (include_rated=True)
    engagement_map: Dict[int, float] = {}
    try:
        engagement_results = score_all_watched_scenes(limit=history_limit, include_rated=True)
        engagement_map = {r.entity_id: r.score for r in engagement_results}
    except Exception:
        _log.debug("content_based: engagement scores unavailable, using uniform weights")

    # For rated-but-unwatched scenes with no engagement score, use their rating
    if rated_only_ids:
        rating_scores = _get_rating_scores_for_scenes(rated_only_ids)
        for sid in rated_only_ids:
            if sid not in engagement_map:
                engagement_map[sid] = rating_scores.get(sid, 0.5)

    # Apply recency boost and data-depth weighting
    taste_weights = engagement_map
    if engagement_map and recency_half_life > 0:
        # Fetch tag data early so we can use it for data-depth scoring
        watched_tags = fetch_scene_tag_ids(list(watched_scene_ids), exclude_tag_ids=blacklisted_tag_ids)

        # Hydrate watched scenes for performer/studio extraction
        watched_payloads = fetch_scenes_by_ids(list(watched_scene_ids))
        watched_performers = _extract_scene_performers(watched_payloads)

        depth_scores = compute_data_depth(
            list(watched_scene_ids),
            corpus_tags=corpus,
            performer_map=watched_performers,
        )
        taste_weights = build_taste_weights(
            engagement_map, history,
            half_life_days=recency_half_life,
            data_depth=depth_scores,
        )
    else:
        watched_tags = fetch_scene_tag_ids(list(watched_scene_ids), exclude_tag_ids=blacklisted_tag_ids)
        watched_payloads = fetch_scenes_by_ids(list(watched_scene_ids))
        watched_performers = _extract_scene_performers(watched_payloads)

    # Build user tag profile
    user_profile = build_user_tag_profile(
        watched_scene_tags=watched_tags,
        idf=idf,
        engagement_scores=taste_weights if taste_weights else None,
    )

    if not user_profile and embedding_weight <= 0 and ai_tag_weight <= 0 and performer_weight <= 0:
        _log.info("content_based: could not build user tag profile (no tagged watched scenes)")
        return {"scenes": [], "total": 0, "has_more": False}

    watched_studios = _extract_scene_studio_ids(watched_payloads)

    performer_profile = build_user_performer_profile(
        watched_scene_performers=watched_performers,
        engagement_scores=taste_weights if taste_weights else None,
    )
    studio_affinity = _build_studio_affinity(watched_studios, taste_weights if taste_weights else None)

    # ------------------------------------------------------------------
    # Step 2b: Build negative profiles from disliked content
    # ------------------------------------------------------------------
    negative_tag_profile: Dict[int, float] = {}
    negative_performer_prof: Dict[int, float] = {}
    if negative_weight > 0 and engagement_map:
        liked_ids, disliked_ids = split_by_engagement(
            list(watched_scene_ids), engagement_map,
        )
        if disliked_ids:
            disliked_tags = {sid: watched_tags.get(sid, set()) for sid in disliked_ids if sid in watched_tags}
            liked_tags = {sid: watched_tags.get(sid, set()) for sid in liked_ids if sid in watched_tags}
            negative_tag_profile = build_negative_tag_profile(
                disliked_scene_tags=disliked_tags,
                liked_scene_tags=liked_tags,
                idf=idf,
            )
            disliked_performers = {sid: watched_performers.get(sid, set()) for sid in disliked_ids if sid in watched_performers}
            liked_performers_map = {sid: watched_performers.get(sid, set()) for sid in liked_ids if sid in watched_performers}
            negative_performer_prof = build_negative_performer_profile(
                disliked_scene_performers=disliked_performers,
                liked_scene_performers=liked_performers_map,
            )
            if negative_tag_profile:
                _log.info(
                    "content_based: negative profile has %d tags, %d performers",
                    len(negative_tag_profile), len(negative_performer_prof),
                )

    # ------------------------------------------------------------------
    # Step 3: Score candidate scenes (everything NOT watched)
    # ------------------------------------------------------------------
    candidate_ids = [sid for sid in corpus if sid not in watched_scene_ids]
    if not candidate_ids:
        return {"scenes": [], "total": 0, "has_more": False}

    # Build TF-IDF vectors for all candidates
    candidate_tags = {sid: corpus[sid] for sid in candidate_ids if sid in corpus}
    candidate_vectors = build_tfidf_vectors(candidate_tags, idf)

    # ------------------------------------------------------------------
    # Step 3b: Compute embedding similarity (prefer cached centroids)
    # ------------------------------------------------------------------
    embedding_sim_map: Dict[int, float] = {}
    if embedding_weight > 0:
        try:
            # Try precomputed centroids first (fast — no on-the-fly centroid calc)
            embedding_sim_map = find_similar_by_cached_centroids(
                exclude_scene_ids=watched_scene_ids,
            )
            if embedding_sim_map:
                _log.info(
                    "content_based: using cached centroids, %d candidates",
                    len(embedding_sim_map),
                )
        except Exception:
            _log.debug("content_based: cached centroid lookup failed", exc_info=True)

        if not embedding_sim_map:
            try:
                embedding_sim_map = compute_embedding_similarity(
                    watched_scene_ids=list(watched_scene_ids),
                    engagement_scores=taste_weights if taste_weights else None,
                    exclude_scene_ids=watched_scene_ids,
                    mode="taste",
                )
                _log.info(
                    "content_based: embedding similarity found %d candidates",
                    len(embedding_sim_map),
                )
            except Exception:
                _log.debug("content_based: embedding similarity unavailable", exc_info=True)

    # ------------------------------------------------------------------
    # Step 3c: Compute AI tag temporal similarity (duration-weighted)
    # ------------------------------------------------------------------
    ai_tag_sim_map: Dict[int, float] = {}
    if ai_tag_weight > 0:
        try:
            ai_tag_sim_map = compute_ai_tag_similarity(
                watched_scene_ids=list(watched_scene_ids),
                engagement_scores=taste_weights if taste_weights else None,
                candidate_scene_ids=candidate_ids,
                mode="taste",
            )
            _log.info(
                "content_based: AI tag similarity found %d candidates",
                len(ai_tag_sim_map),
            )
        except Exception:
            _log.debug("content_based: AI tag similarity unavailable", exc_info=True)

    # Score each candidate (tag-only pre-filter, no performer/studio yet)
    # Negative penalty at pre-filter stage uses only tag data (no performer yet)
    def _neg_penalty_tag_only(vec: Mapping) -> float:
        if not negative_tag_profile:
            return 0.0
        from stash_ai_server.recommendations.utils.negative_signals import compute_negative_tag_penalty
        return compute_negative_tag_penalty(vec, negative_tag_profile)

    scored: List[Tuple[int, float, Dict[str, Any]]] = []
    tag_scored_ids: Set[int] = set()
    for scene_id in candidate_ids:
        vec = candidate_vectors.get(scene_id, {})

        score, debug = score_scene_against_profile(
            scene_vector=vec,
            user_profile=user_profile,
            scene_performers=None,  # filled after hydration below
            performer_profile=performer_profile,
            scene_studio_id=None,  # filled after hydration below
            studio_affinity=studio_affinity,
            tag_weight=tag_weight,
            performer_weight=performer_weight,
            studio_weight=studio_weight,
            embedding_score=embedding_sim_map.get(scene_id, 0.0),
            embedding_weight=embedding_weight,
            ai_tag_score=ai_tag_sim_map.get(scene_id, 0.0),
            ai_tag_weight=ai_tag_weight,
            negative_penalty=_neg_penalty_tag_only(vec),
            negative_weight=negative_weight,
        )

        # Quick pre-filter: only keep candidates with some relevance
        if score > 0:
            scored.append((scene_id, score, debug))
            tag_scored_ids.add(scene_id)

    # Add embedding-only candidates not found by tag pre-filter
    if embedding_sim_map:
        for scene_id, emb_sim in embedding_sim_map.items():
            if scene_id in tag_scored_ids or scene_id in watched_scene_ids:
                continue
            if emb_sim <= 0:
                continue
            # These scenes have zero tag similarity but embedding says they're relevant
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
                embedding_score=emb_sim,
                embedding_weight=embedding_weight,
                ai_tag_score=ai_tag_sim_map.get(scene_id, 0.0),
                ai_tag_weight=ai_tag_weight,
                negative_penalty=0.0,
                negative_weight=negative_weight,
            )
            if score > 0:
                scored.append((scene_id, score, debug))
                tag_scored_ids.add(scene_id)
                # Ensure we have tag data for these scenes
                if scene_id not in candidate_tags:
                    candidate_tags[scene_id] = corpus.get(scene_id, set())

    # Add AI-tag-only candidates not found by tag or embedding pre-filter
    if ai_tag_sim_map:
        for scene_id, ai_sim in ai_tag_sim_map.items():
            if scene_id in tag_scored_ids or scene_id in watched_scene_ids:
                continue
            if ai_sim <= 0:
                continue
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
                embedding_score=embedding_sim_map.get(scene_id, 0.0),
                embedding_weight=embedding_weight,
                ai_tag_score=ai_sim,
                ai_tag_weight=ai_tag_weight,
                negative_penalty=0.0,
                negative_weight=negative_weight,
            )
            if score > 0:
                scored.append((scene_id, score, debug))
                tag_scored_ids.add(scene_id)
                if scene_id not in candidate_tags:
                    candidate_tags[scene_id] = corpus.get(scene_id, set())

    # Sort by score, take top candidates for full scoring with performer/studio
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:candidate_limit]

    if not scored:
        return {"scenes": [], "total": 0, "has_more": False}

    # Hydrate the top candidates to get performer/studio data
    top_ids = [s[0] for s in scored]
    candidate_payloads = fetch_scenes_by_ids(top_ids)
    candidate_performer_map = _extract_scene_performers(candidate_payloads)
    candidate_studio_map = _extract_scene_studio_ids(candidate_payloads)

    # Re-score with full performer/studio data + negative penalties
    fully_scored: List[Tuple[int, float, Dict[str, Any]]] = []
    for scene_id, _, _ in scored:
        vec = candidate_vectors.get(scene_id, {})
        neg_pen = compute_combined_negative_penalty(
            scene_vector=vec,
            negative_tag_profile=negative_tag_profile,
            scene_performers=candidate_performer_map.get(scene_id),
            negative_performer_profile=negative_performer_prof if negative_performer_prof else None,
        ) if negative_weight > 0 else 0.0

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
            embedding_score=embedding_sim_map.get(scene_id, 0.0),
            embedding_weight=embedding_weight,
            ai_tag_score=ai_tag_sim_map.get(scene_id, 0.0),
            ai_tag_weight=ai_tag_weight,
            negative_penalty=neg_pen,
            negative_weight=negative_weight,
        )

        fully_scored.append((scene_id, score, debug))

    fully_scored.sort(key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Step 5: Apply config-driven tag filtering (include + exclude)
    # ------------------------------------------------------------------
    if cfg_exclude_tag_ids:
        before = len(fully_scored)
        fully_scored = [
            (sid, sc, dbg) for sid, sc, dbg in fully_scored
            if not (candidate_tags.get(sid, set()) & cfg_exclude_tag_ids)
        ]
        _log.info(
            "content_based: excluded %d scenes via tag_filter exclude",
            before - len(fully_scored),
        )
    if cfg_include_tag_ids:
        before = len(fully_scored)
        fully_scored = [
            (sid, sc, dbg) for sid, sc, dbg in fully_scored
            if (candidate_tags.get(sid, set()) & cfg_include_tag_ids)
        ]
        _log.info(
            "content_based: include filter narrowed from %d to %d scenes",
            before, len(fully_scored),
        )

    total = len(fully_scored)

    # Collect tag names for debug display
    all_tag_ids: Set[int] = set()
    for scene_id, _, _ in fully_scored:
        tags = candidate_tags.get(scene_id, set())
        all_tag_ids.update(tags)
    for tid in user_profile:
        all_tag_ids.add(tid)
    tag_names = fetch_tag_names(all_tag_ids)

    # Paginate
    offset = max(0, request.offset or 0)
    limit = max(1, request.limit or 40)
    page = fully_scored[offset:offset + limit]

    if not page:
        return {"scenes": [], "total": total, "has_more": False}

    # Build results with debug metadata
    results: List[Dict[str, Any]] = []
    for scene_id, score, debug in page:
        payload = candidate_payloads.get(scene_id)
        if payload is None:
            continue

        tags = candidate_tags.get(scene_id, set())
        top_tags = _annotate_top_tags(tags, user_profile, idf, tag_names)

        payload["score"] = round(score, 4)
        payload["debug_meta"] = {
            "content_based": {
                "score": round(score, 4),
                "tag_similarity": debug["tag_similarity"],
                "performer_score": debug["performer_score"],
                "studio_score": debug["studio_score"],
                "embedding_score": debug["embedding_score"],
                "ai_tag_score": debug["ai_tag_score"],
                "negative_penalty": debug.get("negative_penalty", 0.0),
                "matched_performers": debug["matched_performers"],
                "weights": debug["weights"],
                "top_matching_tags": top_tags,
                "profile_tags": len(user_profile),
                "negative_tags": len(negative_tag_profile),
                "recency_half_life": recency_half_life,
                "scene_tags": len(tags),
                "corpus_size": total + len(watched_scene_ids),
            },
        }
        results.append(payload)

    has_more = offset + len(page) < total
    return {"scenes": results, "total": total, "has_more": has_more}


# ---------------------------------------------------------------------------
# similar_scene: recommend based on seed scene similarity
# ---------------------------------------------------------------------------

async def _handle_similar_scene(
    *,
    seed_ids: List[int],
    corpus: Dict[int, Set[int]],
    idf: Dict[int, float],
    df_map: Dict[int, int],
    tag_weight: float,
    performer_weight: float,
    studio_weight: float,
    embedding_weight: float,
    ai_tag_weight: float,
    negative_weight: float,
    candidate_limit: int,
    blacklisted_tag_ids: Set[int],
    cfg_include_tag_ids: Set[int],
    cfg_exclude_tag_ids: Set[int],
    request: RecommendationRequest,
) -> Dict[str, Any]:
    # Build seed profile from the seed scenes' tags
    seed_tags = fetch_scene_tag_ids(seed_ids, exclude_tag_ids=blacklisted_tag_ids)
    if not seed_tags and embedding_weight <= 0 and ai_tag_weight <= 0 and performer_weight <= 0:
        return {"scenes": [], "total": 0, "has_more": False}

    # Merge all seed tags into one profile vector (equal weight per seed)
    seed_profile = build_user_tag_profile(
        watched_scene_tags=seed_tags,
        idf=idf,
        engagement_scores=None,
    ) if seed_tags else {}

    if not seed_profile and embedding_weight <= 0 and ai_tag_weight <= 0 and performer_weight <= 0:
        return {"scenes": [], "total": 0, "has_more": False}

    # Hydrate seeds for performer/studio data
    seed_payloads = fetch_scenes_by_ids(seed_ids)
    seed_performers = _extract_scene_performers(seed_payloads)
    seed_studios = _extract_scene_studio_ids(seed_payloads)

    # For similar_scene, performer_profile is just the seed performers with equal weight
    all_seed_performers: Set[int] = set()
    for pids in seed_performers.values():
        all_seed_performers.update(pids)
    performer_profile = {pid: 1.0 for pid in all_seed_performers}

    # Studio affinity from seeds
    seed_studio_set: Set[int] = set(seed_studios.values())
    studio_affinity = {sid: 1.0 for sid in seed_studio_set}

    # ------------------------------------------------------------------
    # Embedding similarity for seed scenes
    # ------------------------------------------------------------------
    seed_set = set(seed_ids)
    embedding_sim_map: Dict[int, float] = {}
    if embedding_weight > 0:
        try:
            embedding_sim_map = compute_embedding_similarity(
                scene_ids=seed_ids,
                exclude_scene_ids=seed_set,
                mode="seed",
            )
            _log.info(
                "content_based similar: embedding found %d candidates",
                len(embedding_sim_map),
            )
        except Exception:
            _log.debug("content_based similar: embedding unavailable", exc_info=True)

    # ------------------------------------------------------------------
    # AI tag temporal similarity for seed scenes
    # ------------------------------------------------------------------
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
            _log.info(
                "content_based similar: AI tag similarity found %d candidates",
                len(ai_tag_sim_map),
            )
        except Exception:
            _log.debug("content_based similar: AI tag similarity unavailable", exc_info=True)

    # Score all non-seed scenes (tag pre-filter)
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

    # Add embedding-only candidates not found by tag pre-filter
    if embedding_sim_map:
        for scene_id, emb_sim in embedding_sim_map.items():
            if scene_id in tag_scored_ids or scene_id in seed_set:
                continue
            if emb_sim <= 0:
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
                embedding_score=emb_sim,
                embedding_weight=embedding_weight,
                ai_tag_score=ai_tag_sim_map.get(scene_id, 0.0),
                ai_tag_weight=ai_tag_weight,
            )
            if score > 0:
                scored.append((scene_id, score, debug))
                tag_scored_ids.add(scene_id)
                if scene_id not in candidate_tags:
                    candidate_tags[scene_id] = corpus.get(scene_id, set())

    # Add AI-tag-only candidates not found by tag or embedding pre-filter
    if ai_tag_sim_map:
        for scene_id, ai_sim in ai_tag_sim_map.items():
            if scene_id in tag_scored_ids or scene_id in seed_set:
                continue
            if ai_sim <= 0:
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
                embedding_score=embedding_sim_map.get(scene_id, 0.0),
                embedding_weight=embedding_weight,
                ai_tag_score=ai_sim,
                ai_tag_weight=ai_tag_weight,
            )
            if score > 0:
                scored.append((scene_id, score, debug))
                tag_scored_ids.add(scene_id)
                if scene_id not in candidate_tags:
                    candidate_tags[scene_id] = corpus.get(scene_id, set())

    scored.sort(key=lambda x: x[1], reverse=True)
    scored = scored[:candidate_limit]

    if not scored:
        return {"scenes": [], "total": 0, "has_more": False}

    # Hydrate top candidates for performer/studio re-scoring
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

    # Apply config-driven tag filtering (include + exclude)
    if cfg_exclude_tag_ids:
        before = len(fully_scored)
        fully_scored = [
            (sid, sc, dbg) for sid, sc, dbg in fully_scored
            if not (candidate_tags.get(sid, set()) & cfg_exclude_tag_ids)
        ]
        _log.info(
            "content_based similar: excluded %d scenes via tag_filter exclude",
            before - len(fully_scored),
        )
    if cfg_include_tag_ids:
        before = len(fully_scored)
        fully_scored = [
            (sid, sc, dbg) for sid, sc, dbg in fully_scored
            if (candidate_tags.get(sid, set()) & cfg_include_tag_ids)
        ]
        _log.info(
            "content_based similar: include filter narrowed from %d to %d scenes",
            before, len(fully_scored),
        )

    total = len(fully_scored)

    # Tag names for debug
    all_tag_ids: Set[int] = set()
    for scene_id, _, _ in fully_scored:
        tags = candidate_tags.get(scene_id, set())
        all_tag_ids.update(tags)
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
        top_tags = _annotate_top_tags(tags, seed_profile, idf, tag_names)

        payload["score"] = round(score, 4)
        payload["debug_meta"] = {
            "content_based": {
                "score": round(score, 4),
                "tag_similarity": debug["tag_similarity"],
                "performer_score": debug["performer_score"],
                "studio_score": debug["studio_score"],
                "embedding_score": debug["embedding_score"],
                "ai_tag_score": debug["ai_tag_score"],
                "matched_performers": debug["matched_performers"],
                "weights": debug["weights"],
                "top_matching_tags": top_tags,
                "seed_scene_ids": seed_ids,
                "scene_tags": len(tags),
                "corpus_size": len(corpus),
            },
        }
        results.append(payload)

    has_more = offset + len(page) < total
    return {"scenes": results, "total": total, "has_more": has_more}
