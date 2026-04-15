"""Engagement-scored recommendation plugin.

Uses the shared engagement scoring foundation to rank scenes by how much
the user has engaged with them.  Serves two purposes:

1. **global_feed** context: surfaces the user's top-engaged scenes for
   review (the debug use-case — "show me what the system thinks I liked").

2. **similar_scene** context: given seed scenes, finds other scenes the
   user has engaged with and returns them ranked by engagement score.
   Future phases will layer embedding similarity on top.

This is the first plugin built on the engagement scoring infrastructure
and is primarily useful for verifying that scores align with intuition.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from stash_ai_server.recommendations.engagement.scorer import (
    score_all_watched_scenes,
    score_scenes,
)
from stash_ai_server.recommendations.models import RecContext, RecommendationRequest
from stash_ai_server.recommendations.registry import recommender
from stash_ai_server.recommendations.utils.scene_fetch import fetch_scenes_by_ids

_log = logging.getLogger(__name__)


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


@recommender(
    id="engagement_scored",
    label="Engagement Scored",
    description=(
        "Ranks scenes by a multi-signal engagement score combining ratings, "
        "o-counts, watch history, rewatch patterns, and recency. "
        "Shows full signal breakdown in debug metadata for tuning."
    ),
    contexts=[RecContext.global_feed, RecContext.similar_scene],
    config=[
        {
            "name": "min_confidence",
            "label": "Min Confidence",
            "type": "slider",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "help": "Minimum signal confidence required. Higher = only scenes with lots of data.",
        },
        {
            "name": "candidate_limit",
            "label": "Max Candidates",
            "type": "number",
            "default": 200,
            "min": 20,
            "max": 1000,
        },
        {
            "name": "w_explicit_rating",
            "label": "Weight: Explicit Rating",
            "type": "slider",
            "default": 1.0,
            "min": 0.0,
            "max": 3.0,
            "step": 0.1,
        },
        {
            "name": "w_stash_o_counter",
            "label": "Weight: O-Counter (Stash)",
            "type": "slider",
            "default": 1.0,
            "min": 0.0,
            "max": 3.0,
            "step": 0.1,
        },
        {
            "name": "w_derived_o_count",
            "label": "Weight: O-Count (Derived)",
            "type": "slider",
            "default": 1.0,
            "min": 0.0,
            "max": 3.0,
            "step": 0.1,
        },
        {
            "name": "w_watch_completeness",
            "label": "Weight: Watch Completeness",
            "type": "slider",
            "default": 1.0,
            "min": 0.0,
            "max": 3.0,
            "step": 0.1,
        },
        {
            "name": "w_watch_duration_ratio",
            "label": "Weight: Watch Duration Ratio",
            "type": "slider",
            "default": 1.0,
            "min": 0.0,
            "max": 3.0,
            "step": 0.1,
        },
        {
            "name": "w_rewatch_sessions",
            "label": "Weight: Rewatch Sessions",
            "type": "slider",
            "default": 1.0,
            "min": 0.0,
            "max": 3.0,
            "step": 0.1,
        },
        {
            "name": "w_play_count",
            "label": "Weight: Play Count",
            "type": "slider",
            "default": 1.0,
            "min": 0.0,
            "max": 3.0,
            "step": 0.1,
        },
    ],
    supports_pagination=True,
    exposes_scores=True,
    needs_seed_scenes=False,
    allows_multi_seed=True,
)
async def engagement_scored(ctx: Dict[str, Any], request: RecommendationRequest):
    """Rank scenes by engagement score with full debug breakdown."""

    cfg = request.config or {}

    min_confidence = _coerce_float(cfg.get("min_confidence"), 0.0)
    candidate_limit = _coerce_int(cfg.get("candidate_limit"), 200)

    # Build weight overrides from config
    weight_overrides: Dict[str, float] = {}
    weight_keys = [
        ("w_explicit_rating", "explicit_rating"),
        ("w_stash_o_counter", "stash_o_counter"),
        ("w_derived_o_count", "derived_o_count"),
        ("w_watch_completeness", "watch_completeness"),
        ("w_watch_duration_ratio", "watch_duration_ratio"),
        ("w_rewatch_sessions", "rewatch_sessions"),
        ("w_play_count", "play_count"),
    ]
    for config_key, signal_name in weight_keys:
        val = cfg.get(config_key)
        if val is not None:
            weight_overrides[signal_name] = _coerce_float(val, 1.0)

    # Score all scenes that have engagement data
    scored_results = score_all_watched_scenes(
        weight_overrides=weight_overrides if weight_overrides else None,
        limit=candidate_limit,
        min_confidence=min_confidence,
    )

    if not scored_results:
        return {"scenes": [], "total": 0, "has_more": False}

    # Filter to seed-related if in similar_scene context
    seed_ids = set(int(s) for s in request.seedSceneIds or [] if s is not None)
    if request.context == RecContext.similar_scene and seed_ids:
        # Exclude seed scenes themselves
        scored_results = [r for r in scored_results if r.entity_id not in seed_ids]

    total = len(scored_results)

    # Paginate
    offset = max(0, request.offset or 0)
    limit = max(1, request.limit or 40)
    page = scored_results[offset:offset + limit]

    if not page:
        return {"scenes": [], "total": total, "has_more": False}

    # Hydrate scene payloads
    page_ids = [r.entity_id for r in page]
    payloads = fetch_scenes_by_ids(page_ids)

    results: List[Dict[str, Any]] = []
    for engagement in page:
        payload = payloads.get(engagement.entity_id)
        if payload is None:
            continue

        payload["score"] = round(engagement.score, 4)

        # Build debug metadata with full signal breakdown
        signal_breakdown = {}
        for name, sv in engagement.signals.items():
            signal_breakdown[name] = {
                "value": round(sv.value, 4) if sv.available else None,
                "raw": sv.raw,
                "available": sv.available,
                "reliability": sv.reliability,
                "weight": sv.weight,
            }

        payload["debug_meta"] = {
            "engagement_scored": {
                "score": round(engagement.score, 4),
                "confidence": round(engagement.confidence, 4),
                "signal_count": engagement.signal_count,
                "total_possible": engagement.total_possible,
                "signals": signal_breakdown,
            }
        }
        results.append(payload)

    has_more = offset + len(page) < total
    return {"scenes": results, "total": total, "has_more": has_more}
