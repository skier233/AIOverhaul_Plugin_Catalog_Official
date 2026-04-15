"""Semantic Visual Search recommender plugin.

Lets users describe what they want to see in natural language and finds
scenes whose MetaCLIP2 visual embeddings are closest in the shared
text-image embedding space.

Supports:
  - Single query  → "red dress"
  - Multi-term    → "blue dress, brunette, blowjob"  (comma-separated)
    Each term is encoded separately and the scene's score is the average
    of its best match per term — so scenes matching ALL terms rank highest.

Scoring per scene:
  ``final = best_section_similarity * (1 + coverage_bonus * section_ratio)``
  where ``section_ratio`` = fraction of a scene's sections that exceed
  a similarity threshold.  This rewards scenes where the concept appears
  across many sections, not just a single frame.

Requires ``transformers`` and ``torch`` for the CLIP text encoder.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sqlalchemy import select, func as sa_func

from stash_ai_server.db.session import get_session_local
from stash_ai_server.models.entity_embeddings import EntityEmbedding
from stash_ai_server.recommendations.models import RecContext, RecommendationRequest
from stash_ai_server.recommendations.registry import recommender
from stash_ai_server.recommendations.utils.scene_fetch import fetch_scenes_by_ids
from stash_ai_server.recommendations.utils.stash_tags import (
    fetch_scene_tag_ids,
    resolve_blacklisted_tag_ids,
)

_log = logging.getLogger(__name__)

VISUAL_PREFIX = "visual_metaclip2"
SECTION_OVERFETCH = 6
# Cosine similarity above this counts as a "matching section"
SECTION_MATCH_THRESHOLD = 0.18


# ---------------------------------------------------------------------------
# pgvector search
# ---------------------------------------------------------------------------

def _search_visual_single(
    query_vector: list[float],
    limit: int = 40,
    coverage_bonus: float = 0.3,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    """Search for a single query vector.

    Returns [(scene_id, score, debug_meta), ...] with coverage-adjusted scores.
    """
    with get_session_local()() as session:
        dist_expr = EntityEmbedding.embedding.cosine_distance(query_vector)
        q = (
            select(
                EntityEmbedding.entity_id,
                dist_expr.label("distance"),
                EntityEmbedding.embedding_type,
            )
            .where(
                EntityEmbedding.embedding_type.like(VISUAL_PREFIX + "%"),
                EntityEmbedding.entity_type == "scene",
            )
            .order_by(dist_expr)
            .limit(limit * SECTION_OVERFETCH)
        )
        rows = session.execute(q).all()

    # Collect per-scene section data
    scene_sections: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        sim = max(0.0, 1.0 - float(r.distance))
        scene_sections[r.entity_id].append(sim)

    # Count total sections per scene (need this for coverage ratio)
    scene_total_sections: Dict[int, int] = {}
    if scene_sections:
        with get_session_local()() as session:
            counts = session.execute(
                select(
                    EntityEmbedding.entity_id,
                    sa_func.count().label("cnt"),
                )
                .where(
                    EntityEmbedding.embedding_type.like(VISUAL_PREFIX + "%"),
                    EntityEmbedding.entity_type == "scene",
                    EntityEmbedding.entity_id.in_(list(scene_sections.keys())),
                )
                .group_by(EntityEmbedding.entity_id)
            ).all()
            for row in counts:
                scene_total_sections[row.entity_id] = row.cnt

    results: List[Tuple[int, float, Dict[str, Any]]] = []
    for sid, sims in scene_sections.items():
        best_sim = max(sims)
        total = scene_total_sections.get(sid, len(sims))
        matching = sum(1 for s in sims if s >= SECTION_MATCH_THRESHOLD)
        # Use total sections from DB, with matching from retrieved rows
        coverage_ratio = matching / max(total, 1)
        score = best_sim * (1.0 + coverage_bonus * coverage_ratio)

        debug = {
            "best_section_sim": round(best_sim, 4),
            "matching_sections": matching,
            "total_sections": total,
            "coverage_ratio": round(coverage_ratio, 3),
            "final_score": round(score, 4),
        }
        results.append((sid, score, debug))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]


def _search_multi_term(
    query_terms: List[str],
    encode_fn,
    limit: int = 40,
    coverage_bonus: float = 0.3,
) -> List[Tuple[int, float, Dict[str, Any]]]:
    """Search for multiple terms and combine scores.

    For each term, runs a per-vector search.  The final score for a scene
    is the mean of its best scores across all terms — so scenes matching
    ALL terms rank highest.
    """
    term_results: List[Dict[int, Tuple[float, Dict]]] = []
    all_scene_ids: Set[int] = set()

    for term in query_terms:
        vec = encode_fn(term)
        if vec is None:
            continue
        hits = _search_visual_single(vec, limit=limit * 2, coverage_bonus=coverage_bonus)
        term_map: Dict[int, Tuple[float, Dict]] = {}
        for sid, score, debug in hits:
            term_map[sid] = (score, debug)
            all_scene_ids.add(sid)
        term_results.append(term_map)

    if not term_results:
        return []

    # Combine: for each scene, its score is the MEAN of per-term scores.
    # If a scene doesn't appear in a term's results, it gets 0 for that term.
    combined: List[Tuple[int, float, Dict[str, Any]]] = []
    n_terms = len(term_results)

    for sid in all_scene_ids:
        term_scores = []
        term_debugs = []
        for i, tmap in enumerate(term_results):
            if sid in tmap:
                score, debug = tmap[sid]
                term_scores.append(score)
                term_debugs.append({
                    "term": query_terms[i] if i < len(query_terms) else "?",
                    **debug,
                })
            else:
                term_scores.append(0.0)
                term_debugs.append({"term": query_terms[i] if i < len(query_terms) else "?", "final_score": 0.0})

        # Mean across all terms
        avg_score = sum(term_scores) / n_terms
        # Bonus: how many terms actually matched (non-zero)
        match_count = sum(1 for s in term_scores if s > 0)
        match_ratio = match_count / n_terms
        # Boost scenes that match ALL terms
        final_score = avg_score * (0.5 + 0.5 * match_ratio)

        combined.append((sid, final_score, {
            "multi_term": True,
            "terms": term_debugs,
            "avg_score": round(avg_score, 4),
            "terms_matched": match_count,
            "total_terms": n_terms,
            "match_ratio": round(match_ratio, 3),
        }))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:limit]


# ---------------------------------------------------------------------------
# Recommender entry point
# ---------------------------------------------------------------------------

@recommender(
    id="semantic_search",
    label="Semantic Visual Search",
    description=(
        "Search your library by describing what you want to see. "
        "Uses MetaCLIP2 visual embeddings to match natural-language queries "
        "against scene content.  Supports comma-separated multi-term search "
        "(e.g. 'blue dress, brunette, blowjob').  Requires transformers + torch."
    ),
    contexts=[RecContext.global_feed],
    config=[
        {
            "name": "query",
            "label": "Search",
            "type": "search",
            "default": "",
            "required": True,
            "help": "Describe visuals. Use commas for multi-term: 'red dress, brunette, kitchen'",
            "persist": False,
        },
        {
            "name": "result_limit",
            "label": "Max Results",
            "type": "slider",
            "default": 40,
            "min": 10,
            "max": 200,
            "step": 10,
        },
        {
            "name": "coverage_bonus",
            "label": "Coverage Bonus",
            "type": "slider",
            "default": 0.3,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "How much to reward scenes where the concept appears across many sections",
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
)
async def handle_semantic_search(
    settings: dict,
    request: RecommendationRequest,
) -> Dict[str, Any]:
    cfg = request.config or {}
    query = (cfg.get("query") or "").strip()
    if not query:
        return {"scenes": [], "total": 0, "has_more": False}

    # Lazy-import so the plugin still registers even without torch
    from .text_encoder import encode_text_query, is_available

    if not is_available():
        _log.warning("semantic_search: transformers/torch not installed")
        return {
            "scenes": [],
            "total": 0,
            "has_more": False,
            "error": "Text encoder unavailable — install transformers and torch",
        }

    result_limit = int(cfg.get("result_limit", 40))
    coverage_bonus = float(cfg.get("coverage_bonus", 0.3))

    # Parse comma-separated terms
    terms = [t.strip() for t in query.split(",") if t.strip()]

    if len(terms) > 1:
        # Multi-term search
        scored = _search_multi_term(
            terms,
            encode_fn=encode_text_query,
            limit=result_limit,
            coverage_bonus=coverage_bonus,
        )
    else:
        # Single term search
        query_vector = encode_text_query(terms[0] if terms else query)
        if query_vector is None:
            return {
                "scenes": [],
                "total": 0,
                "has_more": False,
                "error": "Text encoder failed to load",
            }
        scored = _search_visual_single(
            query_vector,
            limit=result_limit,
            coverage_bonus=coverage_bonus,
        )

    if not scored:
        return {"scenes": [], "total": 0, "has_more": False}

    # ── Tag filter ──
    _tag_filter_cfg = cfg.get("tag_filter") or {}
    cfg_include_tag_ids: Set[int] = set()
    cfg_exclude_tag_ids: Set[int] = set()
    if isinstance(_tag_filter_cfg, dict):
        cfg_include_tag_ids = set(_tag_filter_cfg.get("include", []))
        cfg_exclude_tag_ids = set(_tag_filter_cfg.get("exclude", []))

    if cfg_include_tag_ids or cfg_exclude_tag_ids:
        all_scene_ids = [sid for sid, _, _ in scored]
        blacklisted = resolve_blacklisted_tag_ids()
        scene_tags = fetch_scene_tag_ids(all_scene_ids, exclude_tag_ids=blacklisted)

        if cfg_exclude_tag_ids:
            before = len(scored)
            scored = [
                (sid, sc, dbg) for sid, sc, dbg in scored
                if not (scene_tags.get(sid, set()) & cfg_exclude_tag_ids)
            ]
            _log.info("semantic_search: excluded %d scenes via tag_filter exclude", before - len(scored))

        if cfg_include_tag_ids:
            before = len(scored)
            scored = [
                (sid, sc, dbg) for sid, sc, dbg in scored
                if (scene_tags.get(sid, set()) & cfg_include_tag_ids)
            ]
            _log.info("semantic_search: include filter narrowed from %d to %d scenes", before, len(scored))

    total = len(scored)

    # Paginate
    offset = max(0, request.offset or 0)
    limit = max(1, request.limit or 40)
    page = scored[offset:offset + limit]

    if not page:
        return {"scenes": [], "total": total, "has_more": False}

    # Hydrate scene payloads
    scene_ids = [sid for sid, _, _ in page]
    payloads = fetch_scenes_by_ids(scene_ids)

    results: List[Dict[str, Any]] = []
    for scene_id, score, debug in page:
        payload = payloads.get(scene_id)
        if payload is None:
            continue

        payload["score"] = round(score, 4)
        payload["debug_meta"] = {
            "semantic_search": {
                "score": round(score, 4),
                "query": query,
                **debug,
            },
        }
        results.append(payload)

    has_more = offset + len(page) < total
    return {"scenes": results, "total": total, "has_more": has_more}
