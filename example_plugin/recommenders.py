from stash_ai_server.recommendations.registry import recommender
from stash_ai_server.recommendations.models import RecContext, RecommendationRequest
from stash_ai_server.utils.stash_api import stash_api

@recommender(id='example_plugin.random', label='Example Random', contexts=[RecContext.global_feed])
async def example_random(ctx: dict, req: RecommendationRequest):
    """Return an empty list (demo recommender). ctx currently unused."""

    
    cfg = req.config or {}
    limit = req.limit or 40
    offset = req.offset or 0

    # We purposely rely on API layer for offset semantics; handler returns already-sliced page.
    # Tag id 118 = sample corpus anchor; future: replace with popularity index query.
    scenes, approx_total, has_more = stash_api.fetch_scenes_by_tag_paginated(1508, offset, limit)
    return {
        'scenes': scenes,
        'total': approx_total,
        'has_more': has_more
    }

