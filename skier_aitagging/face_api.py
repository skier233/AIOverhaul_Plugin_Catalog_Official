"""API endpoints for face cluster management.

Mounted at ``/api/v1/plugins/skier_aitagging/faces/`` by the plugin service.
"""
from __future__ import annotations

import asyncio
import io
import logging
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from stash_ai_server.api.plugins import _require_plugin_active
from stash_ai_server.db.detection_store import (
    delete_cluster,
    delete_cluster_async,
    delete_clusters_bulk,
    delete_performer_assignments_for_cluster,
    find_nearest_cluster_async,
    get_bulk_cluster_entity_pairs,
    get_cluster_by_id,
    get_cluster_embeddings,
    get_cluster_exemplars,
    get_cluster_entity_pairs,
    get_cluster_for_performer,
    get_clusters_by_ids,
    get_entity_count_by_type,
    get_entity_tracks,
    get_orphaned_performer_entities,
    link_performer,
    list_cluster_ids,
    list_clusters,
    list_clusters_async,
    merge_clusters,
    record_performer_assignments,
    remove_exemplar_from_cluster,
)
from stash_ai_server.db.session import get_db, get_session_local
from stash_ai_server.models.detections import FaceCluster
from stash_ai_server.models.ratings import EntityRating
from stash_ai_server.utils import stash_db
from stash_ai_server.utils.stash_api import stash_api

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/skier_aitagging/faces", tags=["skier_aitagging_faces"])

PLUGIN_NAME = "skier_aitagging"


def _apply_performer_to_cluster_entities(
    cluster_id: int,
    performer_id: int,
) -> dict[str, int]:
    """Apply a performer to all Stash scenes/images where *cluster_id* appears.

    Before adding, checks which entities already have the performer in Stash.
    Only records an assignment in ``face_performer_assignments`` for entities
    where the performer was NOT already present, so we never remove a
    performer we didn't originally add.

    Returns ``{"scenes_updated": N, "images_updated": M}``.
    """
    entity_pairs = get_cluster_entity_pairs(cluster_id)
    scene_ids = [eid for etype, eid in entity_pairs if etype == "scene"]
    image_ids = [eid for etype, eid in entity_pairs if etype == "image"]

    # Determine which entities already have this performer before we touch them
    pre_existing = _get_entities_with_performer(scene_ids, image_ids, performer_id)

    scenes_updated = 0
    images_updated = 0

    if scene_ids:
        try:
            stash_api.add_performer_to_scenes(scene_ids, performer_id)
            scenes_updated = len(scene_ids)
        except Exception:
            _log.exception(
                "Failed to apply performer %s to %d scenes for cluster %s",
                performer_id, len(scene_ids), cluster_id,
            )

    if image_ids:
        try:
            stash_api.add_performer_to_images(image_ids, performer_id)
            images_updated = len(image_ids)
        except Exception:
            _log.exception(
                "Failed to apply performer %s to %d images for cluster %s",
                performer_id, len(image_ids), cluster_id,
            )

    # Record assignments only for entities where the performer was NEW
    new_pairs = [
        (etype, eid) for etype, eid in entity_pairs
        if (etype, eid) not in pre_existing
    ]
    if new_pairs:
        record_performer_assignments(new_pairs, performer_id, cluster_id)
        _log.info(
            "Recorded %d new performer assignment(s) for performer %s via cluster %s",
            len(new_pairs), performer_id, cluster_id,
        )

    if scenes_updated or images_updated:
        _log.info(
            "Retroactively applied performer %s to %d scene(s) + %d image(s) for cluster %s",
            performer_id, scenes_updated, images_updated, cluster_id,
        )

    return {"scenes_updated": scenes_updated, "images_updated": images_updated}


def _get_entities_with_performer(
    scene_ids: list[int],
    image_ids: list[int],
    performer_id: int,
) -> set[tuple[str, int]]:
    """Query Stash SQLite DB to find which entities already have *performer_id*.

    Returns a set of ``(entity_type, entity_id)`` tuples where the performer
    was already present before any AI assignment.
    """
    pre_existing: set[tuple[str, int]] = set()
    if not scene_ids and not image_ids:
        return pre_existing

    session_factory = stash_db.get_stash_sessionmaker()
    if session_factory is None:
        # Can't check — conservatively assume none are pre-existing
        # so we record all assignments (safe: we may skip removal later
        # but never destructively remove a pre-existing performer)
        return pre_existing

    try:
        with session_factory() as session:
            if scene_ids:
                scene_link = stash_db.get_first_available_table(
                    "performers_scenes", "scene_performers", "performer_scenes",
                    required_columns=("scene_id", "performer_id"),
                )
                if scene_link is not None:
                    s_col = scene_link.c.get("scene_id")
                    p_col = scene_link.c.get("performer_id")
                    if s_col is not None and p_col is not None:
                        rows = session.execute(
                            sa.select(s_col)
                            .where(s_col.in_(scene_ids), p_col == performer_id)
                        ).all()
                        for row in rows:
                            pre_existing.add(("scene", int(row[0])))

            if image_ids:
                image_link = stash_db.get_first_available_table(
                    "performers_images", "image_performers", "performer_images",
                    required_columns=("image_id", "performer_id"),
                )
                if image_link is not None:
                    i_col = image_link.c.get("image_id")
                    p_col = image_link.c.get("performer_id")
                    if i_col is not None and p_col is not None:
                        rows = session.execute(
                            sa.select(i_col)
                            .where(i_col.in_(image_ids), p_col == performer_id)
                        ).all()
                        for row in rows:
                            pre_existing.add(("image", int(row[0])))
    except Exception:
        _log.debug("Failed to check pre-existing performers in Stash DB", exc_info=True)

    return pre_existing


def _resolve_performer_name(payload: "LinkPerformerRequest") -> str | None:
    """Return the performer's display name to store as the cluster label.

    Uses the name sent by the client if present, otherwise falls back to
    a Stash SQLite lookup so the name is always stored regardless of which
    UI path triggered the link.
    """
    if payload.performer_name:
        return payload.performer_name.strip() or None

    # Fallback: query Stash SQLite directly
    performers_table = stash_db.get_stash_table("performers", required=False)
    if performers_table is None:
        return None
    session_factory = stash_db.get_stash_sessionmaker()
    if session_factory is None:
        return None
    try:
        with session_factory() as session:
            perf_id_col = performers_table.c.get("id")
            perf_name_col = performers_table.c.get("name")
            if perf_id_col is None or perf_name_col is None:
                return None
            row = session.execute(
                sa.select(perf_name_col).where(perf_id_col == payload.performer_id).limit(1)
            ).fetchone()
            return row[0] if row else None
    except Exception:
        _log.debug("Could not resolve performer name for id=%s", payload.performer_id)
        return None


async def _set_performer_image_from_cluster(
    cluster_id: int,
    performer_id: int,
    size: int = 400,
    pad: float = 0.2,
    thumbnail_index: int | None = None,
) -> None:
    """Generate a face crop from the cluster's best exemplar and push it
    to Stash as the performer's image via GraphQL."""
    import base64

    candidates = _get_exemplar_candidates(cluster_id, limit=5)
    if not candidates:
        return

    # Pick the sharpest candidate
    best_source: bytes | None = None
    best_bbox: list[float] | None = None
    best_sharpness: float = -1.0

    video_path: str | None = None
    first_entity_type = candidates[0][0]
    first_entity_id = candidates[0][1]
    if first_entity_type == "scene":
        video_path = await asyncio.to_thread(_get_scene_file_path, first_entity_id)
        if video_path and not Path(video_path).is_file():
            video_path = None

    for entity_type, entity_id, bbox, timestamp_s in candidates:
        if not bbox or len(bbox) != 4:
            continue
        source_bytes: bytes | None = None
        if entity_type == "scene" and video_path:
            source_bytes = await asyncio.to_thread(
                _extract_video_frame, video_path, timestamp_s,
            )
        if source_bytes is None:
            source_bytes = await _fetch_source_via_stash(entity_type, entity_id)
        if source_bytes is None:
            continue
        sharpness = await asyncio.to_thread(_face_sharpness, source_bytes, bbox)
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_source = source_bytes
            best_bbox = bbox

    # If a specific thumbnail_index was requested, try to use the cached
    # thumbnail file directly (already ranked by sharpness).  If the cache
    # doesn't exist, fall through to the sharpest (index 0) logic above.
    if thumbnail_index is not None and thumbnail_index >= 0:
        cache_dir = _get_thumbnail_cache_dir()
        pad_key = int(pad * 100)
        idx_cache = cache_dir / f"{cluster_id}_{size}_i{thumbnail_index}_p{pad_key}.jpg"
        if not idx_cache.exists():
            # Generate all variants via the thumbnail path so the cache exists
            idx_cache = cache_dir / f"{cluster_id}_400_i{thumbnail_index}_p20.jpg"
        if idx_cache.exists():
            import base64
            b64 = base64.b64encode(idx_cache.read_bytes()).decode("ascii")
            data_uri = f"data:image/jpeg;base64,{b64}"
            stash_api.update_performer_image(performer_id, data_uri)
            _log.debug("Set performer %s image from cluster %d thumbnail index %d (cached)", performer_id, cluster_id, thumbnail_index)
            return
        # If cache miss for the specific index, fall through to re-rank

    if best_source is None or best_bbox is None:
        return

    # Crop
    from PIL import Image

    img = Image.open(io.BytesIO(best_source))
    w, h = img.size
    clamped = [max(0.0, min(1.0, v)) for v in best_bbox]
    x1, y1, x2, y2 = clamped
    if all(0 <= v <= 1.0 for v in clamped):
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    else:
        x1, y1, x2, y2 = int(best_bbox[0]), int(best_bbox[1]), int(best_bbox[2]), int(best_bbox[3])

    pad_w = int((x2 - x1) * pad)
    pad_h = int((y2 - y1) * pad)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    if (x2 - x1) < 1 or (y2 - y1) < 1:
        return

    face_crop = img.crop((x1, y1, x2, y2))
    if face_crop.mode not in ("RGB", "L"):
        face_crop = face_crop.convert("RGB")

    # Scale longest edge to `size`
    crop_w, crop_h = face_crop.size
    if crop_w >= crop_h:
        new_w = size
        new_h = max(1, int(size * crop_h / crop_w))
    else:
        new_h = size
        new_w = max(1, int(size * crop_w / crop_h))
    face_crop = face_crop.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    face_crop.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    data_uri = f"data:image/jpeg;base64,{b64}"

    stash_api.update_performer_image(performer_id, data_uri)
    _log.debug("Set performer %s image from cluster %d", performer_id, cluster_id)


async def _hydrate_performer_from_stashdb_ref(
    performer_id: int,
    ref: StashDBPerformerRef,
) -> None:
    """Attach StashDB metadata to a local performer.

    1. Set the ``stash_ids`` link so Stash knows this performer is on a
       stash-box.
    2. Use Stash's built-in stash-box scraper to pull the **full** profile
       (image, gender, birthdate, ethnicity, country, …) in one shot.
    3. Fall back to the cached ``image_url`` on the ref when the scraper
       is unavailable or the endpoint unreachable.
    """
    # Step 1 — establish the stash_ids link (required for the scraper)
    stash_ids = None
    if ref.source_endpoint and ref.stashdb_id:
        stash_ids = [{"stash_id": ref.stashdb_id, "endpoint": ref.source_endpoint}]

    if stash_ids:
        ok = await stash_api.update_performer_async(
            performer_id, stash_ids=stash_ids,
        )
        if not ok:
            _log.warning("Failed to set stash_ids on performer %s", performer_id)

    # Step 2 — scrape from the stash-box for full data
    if ref.source_endpoint:
        scraped = await stash_api.scrape_performer_from_stashbox_async(
            performer_id, ref.source_endpoint,
        )
        if scraped:
            applied = await _apply_scraped_performer_data(performer_id, scraped)
            if applied:
                return

    # Step 3 — fallback: try image_url stored on the ref
    image_url = getattr(ref, "image_url", None)
    if image_url:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(image_url, timeout=15.0, follow_redirects=True)
                if resp.status_code == 200 and resp.content:
                    import base64
                    ct = resp.headers.get("content-type", "image/jpeg")
                    if "/" not in ct:
                        ct = "image/jpeg"
                    b64 = base64.b64encode(resp.content).decode("ascii")
                    await stash_api.update_performer_image_async(
                        performer_id, f"data:{ct};base64,{b64}",
                    )
                    return
        except Exception:
            _log.debug("Failed to fetch image for performer %s from %s",
                        performer_id, image_url)

    # Step 4 — at minimum set disambiguation if we haven't scraped it
    if ref.disambiguation and not stash_ids:
        await stash_api.update_performer_async(
            performer_id, disambiguation=ref.disambiguation,
        )


async def _apply_scraped_performer_data(
    performer_id: int,
    scraped: Any,
) -> bool:
    """Build a ``PerformerUpdateInput`` from stash-box scraped data and apply it.

    Downloads the performer image from the stash-box CDN if available.
    Returns ``True`` when at least one field was successfully written.
    """
    scraped = _normalize_scraped_performer(scraped)
    if not scraped:
        return False

    update: dict[str, Any] = {"id": performer_id}

    # Simple string fields that map 1-to-1
    for field in (
        "gender", "birthdate", "death_date", "ethnicity", "country",
        "eye_color", "hair_color", "measurements", "fake_tits",
        "career_length", "tattoos", "piercings", "details",
        "disambiguation", "circumcised",
    ):
        val = scraped.get(field)
        if val:
            update[field] = val

    # Height: scraped returns a string, update expects height_cm (int)
    height = scraped.get("height")
    if height:
        try:
            update["height_cm"] = int(height)
        except (ValueError, TypeError):
            pass

    # Weight
    weight = scraped.get("weight")
    if weight:
        try:
            update["weight"] = int(weight)
        except (ValueError, TypeError):
            pass

    # Penis length (float)
    penis_length = scraped.get("penis_length")
    if penis_length:
        try:
            update["penis_length"] = float(penis_length)
        except (ValueError, TypeError):
            pass

    # Aliases (scraped returns newline-separated string)
    aliases = scraped.get("aliases") or scraped.get("alias_list")
    if aliases:
        if isinstance(aliases, str):
            update["alias_list"] = [a.strip() for a in aliases.split("\n") if a.strip()]
        elif isinstance(aliases, list):
            update["alias_list"] = [a.get("name") if isinstance(a, dict) else a for a in aliases]
            update["alias_list"] = [a for a in update["alias_list"] if isinstance(a, str) and a.strip()]

    # URLs
    url = scraped.get("url")
    if url:
        update["url"] = url
    urls = scraped.get("urls")
    if urls and isinstance(urls, list):
        update["urls"] = [
            u["url"] if isinstance(u, dict) else u
            for u in urls
            if (isinstance(u, dict) and u.get("url")) or (isinstance(u, str) and u)
        ]

    # Image — download from the stash-box CDN
    image_set = False
    images = scraped.get("images")
    if images and isinstance(images, list):
        for img_item in images:
            img_url = img_item.get("url") if isinstance(img_item, dict) else img_item
            if not isinstance(img_url, str) or not img_url.startswith("http"):
                continue
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(img_url, timeout=15.0, follow_redirects=True)
                    if resp.status_code == 200 and resp.content:
                        import base64
                        ct = resp.headers.get("content-type", "image/jpeg")
                        if "/" not in ct:
                            ct = "image/jpeg"
                        update["image"] = (
                            f"data:{ct};base64,"
                            f"{base64.b64encode(resp.content).decode('ascii')}"
                        )
                        image_set = True
                        break
            except Exception:
                _log.debug("Failed to fetch stash-box image from %s", img_url)

    if not image_set:
        img = scraped.get("image")
        if img and isinstance(img, str):
            if img.startswith("data:"):
                update["image"] = img
            elif img.startswith("http"):
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(img, timeout=15.0, follow_redirects=True)
                        if resp.status_code == 200 and resp.content:
                            import base64
                            ct = resp.headers.get("content-type", "image/jpeg")
                            if "/" not in ct:
                                ct = "image/jpeg"
                            update["image"] = (
                                f"data:{ct};base64,"
                                f"{base64.b64encode(resp.content).decode('ascii')}"
                            )
                except Exception:
                    _log.debug("Failed to fetch stash-box image from %s", img)

    if len(update) <= 1:  # only "id"
        return False

    ok = await stash_api.full_update_performer_async(update)
    if ok:
        _log.info(
            "Applied stash-box scrape to performer %s (%d field(s))",
            performer_id, len(update) - 1,
        )
    return ok


def _normalize_scraped_performer(scraped: Any) -> dict[str, Any] | None:
    """Return the first usable scraped performer payload as a dict.

    ``stashapi`` returns ``scrapeSinglePerformer`` as a list. Depending on the
    Stash version, nested values like images and aliases may also be dicts.
    """
    if isinstance(scraped, dict):
        if isinstance(scraped.get("performer"), dict):
            return scraped["performer"]
        return scraped

    if isinstance(scraped, list):
        for item in scraped:
            normalized = _normalize_scraped_performer(item)
            if normalized:
                return normalized
        return None

    return None


# Thumbnail cache directory — created lazily
_THUMBNAIL_CACHE_DIR: Path | None = None

# All thumbnails are generated and cached at this single size regardless
# of the ``size`` query parameter.  Different UI contexts (grid=180,
# detail=200, review=150, etc.) all share the same cache entry.
# The browser/CSS handles display scaling.
_CANONICAL_THUMB_SIZE = 300


def _get_thumbnail_cache_dir() -> Path:
    global _THUMBNAIL_CACHE_DIR
    if _THUMBNAIL_CACHE_DIR is None:
        # Prefer Stash's generated files directory so face crops live
        # alongside other Stash-generated content.
        from stash_ai_server.utils.path_mutation import mutate_path_for_backend
        gen_path = stash_api.get_stash_generated_path()
        if gen_path:
            try:
                mutated = mutate_path_for_backend(gen_path)
                p = Path(mutated) / "ai_face_crops"
                p.mkdir(parents=True, exist_ok=True)
                # Verify the dir is actually writable
                test_file = p / ".cache_test"
                test_file.write_bytes(b"ok")
                test_file.unlink()
                _THUMBNAIL_CACHE_DIR = p
                _log.debug("Face thumbnail cache: %s (Stash generated)", p)
                return _THUMBNAIL_CACHE_DIR
            except Exception:
                _log.warning("Cannot use Stash generated path for face crops; falling back", exc_info=True)
        else:
            _log.debug("Stash generated path unavailable; using data_dir for face thumbnails")
        # Fallback to data_dir
        from stash_ai_server.core.config import settings
        _THUMBNAIL_CACHE_DIR = Path(settings.data_dir) / "face_thumbnails"
        _THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _log.debug("Face thumbnail cache (fallback): %s", _THUMBNAIL_CACHE_DIR)
    return _THUMBNAIL_CACHE_DIR


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class LinkPerformerRequest(BaseModel):
    performer_id: int
    performer_name: Optional[str] = None
    set_performer_image: bool = False
    thumbnail_index: int | None = None
    hydrate_from_stashdb: bool = True


class MergeClustersRequest(BaseModel):
    surviving_id: int
    absorbed_id: int


class ClusterSummary(BaseModel):
    id: int
    status: str
    performer_id: Optional[int] = None
    sample_count: int
    quality_score: Optional[float] = None
    label: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class ClusterDetail(BaseModel):
    id: int
    status: str
    performer_id: Optional[int] = None
    sample_count: int
    quality_score: Optional[float] = None
    label: Optional[str] = None
    exemplars: list[dict] = []


class TrackSummary(BaseModel):
    id: int
    label: str
    bbox: list[float]
    score: float
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    cluster_id: Optional[int] = None
    cluster_status: Optional[str] = None
    performer_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Cluster endpoints
# ---------------------------------------------------------------------------

def _get_bulk_cluster_ratings(cluster_ids: list[int]) -> dict[int, int | None]:
    """Return ``{cluster_id: rating100_or_None}`` for the given clusters."""
    if not cluster_ids:
        return {}
    str_ids = [str(cid) for cid in cluster_ids]
    with get_session_local()() as session:
        rows = session.execute(
            sa.select(EntityRating.entity_id, EntityRating.value).where(
                EntityRating.entity_type == "face_cluster",
                EntityRating.entity_id.in_(str_ids),
                EntityRating.rating_key == "default",
            )
        ).all()
        return {int(r.entity_id): r.value for r in rows}

@router.get("/clusters")
async def get_clusters(
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search by performer name (label)"),
    performer_id: Optional[int] = Query(None, description="Filter by linked performer ID"),
    sort: Optional[str] = Query(None, description="Sort column: updated_at, created_at, sample_count, quality_score, label, id, suggestion_confidence, scene_count, image_count"),
    sort_dir: Optional[str] = Query(None, description="Sort direction: asc or desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List face clusters with optional status filter, search, and sort."""
    _require_plugin_active(db, PLUGIN_NAME)
    offset = (page - 1) * per_page

    # Special sort: scene_count / image_count / suggestion_confidence require
    # computing data for ALL matching clusters, sorting in Python, then paginating.
    if sort in ("scene_count", "image_count"):
        target_type = "scene" if sort == "scene_count" else "image"
        all_ids = list_cluster_ids(status=status, search=search, performer_id=performer_id)
        entity_pairs = get_bulk_cluster_entity_pairs(all_ids)
        type_counts: dict[int, int] = {
            cid: sum(1 for et, _ in pairs if et == target_type)
            for cid, pairs in entity_pairs.items()
        }
        ascending = sort_dir == "asc"
        sorted_ids = sorted(all_ids, key=lambda cid: type_counts.get(cid, 0), reverse=not ascending)
        total = len(sorted_ids)
        page_ids = sorted_ids[offset : offset + per_page]
        clusters = get_clusters_by_ids(page_ids)
        suggestions = _compute_batch_top_suggestions(page_ids)
    elif sort == "suggestion_confidence":
        all_ids = list_cluster_ids(status=status, search=search, performer_id=performer_id)
        suggestions = _compute_batch_top_suggestions(all_ids)

        # Bulk-fetch stashdb_match_score for all clusters so it can be
        # factored into the sort.  Uses a single query.
        stashdb_scores: dict[int, float] = {}
        if all_ids:
            with get_session_local()() as _sess:
                _rows = _sess.execute(
                    sa.select(FaceCluster.id, FaceCluster.stashdb_match_score)
                    .where(
                        FaceCluster.id.in_(all_ids),
                        FaceCluster.stashdb_match_score.isnot(None),
                    )
                ).all()
                stashdb_scores = {r[0]: float(r[1]) for r in _rows}

        # Sort: StashDB matches always rank above pure co-occurrence, then
        # by score descending within each tier.
        def _sort_key(cid: int) -> tuple[int, float]:
            stash = stashdb_scores.get(cid, 0.0)
            has_stashdb = 1 if stash > 0 else 0
            co = 0.0
            s = suggestions.get(cid)
            if s:
                co = s["co_occurrence_ratio"]
            score = stash if has_stashdb else co
            return (has_stashdb, score)

        ascending = sort_dir == "asc"
        sorted_ids = sorted(all_ids, key=_sort_key, reverse=not ascending)
        total = len(sorted_ids)
        page_ids = sorted_ids[offset : offset + per_page]
        clusters = get_clusters_by_ids(page_ids)
    elif sort == "rating":
        all_ids = list_cluster_ids(status=status, search=search, performer_id=performer_id)
        ratings_map = _get_bulk_cluster_ratings(all_ids)
        ascending = sort_dir == "asc"
        sorted_ids = sorted(
            all_ids,
            key=lambda cid: (0 if ratings_map.get(cid) is None else 1, ratings_map.get(cid, 0)),
            reverse=not ascending,
        )
        total = len(sorted_ids)
        page_ids = sorted_ids[offset : offset + per_page]
        clusters = get_clusters_by_ids(page_ids)
        suggestions = _compute_batch_top_suggestions(page_ids)
    else:
        clusters, total = list_clusters(
            status=status, search=search, performer_id=performer_id,
            sort=sort, sort_dir=sort_dir,
            offset=offset, limit=per_page,
        )
        suggestions = _compute_batch_top_suggestions([c.id for c in clusters])

    # Enrich each cluster with entity counts, thumbnail URL, and top suggestion
    cluster_ids = [c.id for c in clusters]
    ratings_map = _get_bulk_cluster_ratings(cluster_ids)
    enriched = []
    for c in clusters:
        counts = get_entity_count_by_type(c.id)
        item: dict[str, Any] = {
            "id": c.id,
            "status": c.status,
            "performer_id": c.performer_id,
            "sample_count": c.sample_count,
            "quality_score": c.quality_score,
            "scene_count": counts["scene_count"],
            "image_count": counts["image_count"],
            "label": c.label,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            "thumbnail_url": f"/api/v1/plugins/skier_aitagging/faces/clusters/{c.id}/thumbnail",
        }
        # Include minimal StashDB match info for badge display
        if getattr(c, "stashdb_match_id", None) and getattr(c, "stashdb_match_score", None):
            item["stashdb_match_id"] = c.stashdb_match_id
            item["stashdb_match_score"] = c.stashdb_match_score
        # Rejection state
        item["stashdb_suggestion_rejected"] = bool(getattr(c, "stashdb_suggestion_rejected", False))
        raw_rejected = getattr(c, "rejected_performer_ids", None)
        if raw_rejected:
            import json as _json
            try:
                item["rejected_performer_ids"] = _json.loads(raw_rejected)
            except Exception:
                item["rejected_performer_ids"] = []
        else:
            item["rejected_performer_ids"] = []
        # Include top co-occurrence suggestion
        top_sug = suggestions.get(c.id)
        if top_sug is not None:
            item["top_suggestion"] = top_sug
        # Include default rating if set
        r100 = ratings_map.get(c.id)
        item["rating100"] = r100
        enriched.append(item)

    return {
        "clusters": enriched,
        "total": total,
        "page": page,
        "per_page": per_page,
    }


async def _build_cluster_detail(cluster_id: int) -> dict:
    """Build the full cluster detail dict (shared by GET and DELETE endpoints)."""
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        return None

    exemplars = get_cluster_embeddings(cluster_id, exemplars_only=True)
    _log.debug(
        "_build_cluster_detail cluster %d: %d exemplars, ids=%s",
        cluster_id, len(exemplars), [e.id for e in exemplars],
    )
    counts = get_entity_count_by_type(cluster_id)
    entity_pairs = get_cluster_entity_pairs(cluster_id)

    thumb_base = f"/api/v1/plugins/skier_aitagging/faces/clusters/{cluster_id}/thumbnail"

    # Resolve StashDB match info if present
    stashdb_match_info = None
    if getattr(cluster, "stashdb_match_id", None):
        ref = _get_ref_by_id(cluster.stashdb_match_id)
        if ref:
            verified_local = await _get_verified_local_performer_for_ref(ref, repair=True)
            stashdb_match_info = {
                "ref_id": ref.id,
                "stashdb_id": ref.stashdb_id,
                "name": ref.name,
                "disambiguation": ref.disambiguation,
                "similarity": cluster.stashdb_match_score,
                "source_endpoint": ref.source_endpoint,
                "local_performer_id": verified_local["id"] if verified_local else None,
                "image_url": getattr(ref, "image_url", None),
            }

    return {
        "id": cluster.id,
        "status": cluster.status,
        "performer_id": cluster.performer_id,
        "sample_count": cluster.sample_count,
        "quality_score": cluster.quality_score,
        "label": cluster.label,
        "created_at": cluster.created_at.isoformat() if cluster.created_at else None,
        "updated_at": cluster.updated_at.isoformat() if cluster.updated_at else None,
        "scene_count": counts["scene_count"],
        "image_count": counts["image_count"],
        "thumbnail_url": thumb_base,
        "stashdb_match": stashdb_match_info,
        "entities": [
            {"entity_type": etype, "entity_id": eid}
            for etype, eid in entity_pairs
        ],
        "rating100": _get_bulk_cluster_ratings([cluster_id]).get(cluster_id),
        "exemplars": [
            {
                "id": e.id,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "bbox": list(e.bbox) if e.bbox else None,
                "score": e.score,
                "norm": e.norm,
                "timestamp_s": e.timestamp_s,
                "thumbnail_url": f"{thumb_base}?index={i}&pad=0.2",
            }
            for i, e in enumerate(exemplars)
        ],
    }


@router.get("/clusters/{cluster_id}")
async def get_cluster(cluster_id: int, db: Session = Depends(get_db)):
    """Get cluster detail with exemplar embeddings, entity list, and counts."""
    _require_plugin_active(db, PLUGIN_NAME)
    detail = await _build_cluster_detail(cluster_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    return JSONResponse(
        content=detail,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@router.get("/clusters/{cluster_id}/similar")
async def get_similar_clusters(
    cluster_id: int,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Find face clusters most similar to the given cluster's centroid."""
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    if cluster.centroid is None:
        return {"matches": []}

    centroid = list(cluster.centroid) if not isinstance(cluster.centroid, list) else cluster.centroid
    # Fetch extra so we can exclude the source cluster
    raw = await find_nearest_cluster_async(centroid, limit=limit + 1)
    matches = [(cid, sim) for cid, sim in raw if cid != cluster_id][:limit]
    if not matches:
        return {"matches": []}

    clusters_by_id = {c.id: c for c in get_clusters_by_ids([cid for cid, _ in matches])}
    thumb_base = "/api/v1/plugins/skier_aitagging/faces/clusters"
    result = []
    for cid, sim in matches:
        c = clusters_by_id.get(cid)
        if not c:
            continue
        result.append({
            "id": c.id,
            "label": c.label,
            "status": c.status,
            "performer_id": c.performer_id,
            "sample_count": c.sample_count,
            "quality_score": c.quality_score,
            "similarity": round(sim, 4),
            "thumbnail_url": f"{thumb_base}/{c.id}/thumbnail",
        })
    return {"matches": result}


@router.get("/performers/{performer_id}/similar-by-face")
async def get_similar_performers_by_face(
    performer_id: int,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Find performers whose face clusters are similar to this performer's clusters."""
    _require_plugin_active(db, PLUGIN_NAME)
    # Get all clusters for this performer
    perf_clusters, _ = await list_clusters_async(performer_id=performer_id, limit=100)
    centroids = [c for c in perf_clusters if c.centroid is not None]
    if not centroids:
        return {"matches": []}

    # Search each centroid, gather all similar clusters that belong to OTHER performers
    seen: dict[int, float] = {}  # performer_id -> best similarity
    performer_cluster_map: dict[int, int] = {}  # performer_id -> cluster_id (best)
    exclude_pid = performer_id
    for cluster in centroids:
        centroid = list(cluster.centroid) if not isinstance(cluster.centroid, list) else cluster.centroid
        raw = await find_nearest_cluster_async(centroid, limit=limit * 3)
        for cid, sim in raw:
            c = get_cluster_by_id(cid)
            if not c or not c.performer_id or c.performer_id == exclude_pid:
                continue
            if c.performer_id not in seen or sim > seen[c.performer_id]:
                seen[c.performer_id] = sim
                performer_cluster_map[c.performer_id] = cid

    # Sort by similarity, take top N
    ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:limit]
    if not ranked:
        return {"matches": []}

    thumb_base = "/api/v1/plugins/skier_aitagging/faces/clusters"
    result = []
    for pid, sim in ranked:
        cid = performer_cluster_map[pid]
        c = get_cluster_by_id(cid)
        label = c.label if c else None
        result.append({
            "performer_id": pid,
            "performer_name": label,
            "similarity": round(sim, 4),
            "cluster_id": cid,
            "thumbnail_url": f"{thumb_base}/{cid}/thumbnail" if cid else None,
        })
    return {"matches": result}


@router.get("/scenes/{scene_id}/faces")
async def get_scene_faces(
    scene_id: int,
    db: Session = Depends(get_db),
):
    """Get unique face clusters detected in a scene, with performer info."""
    _require_plugin_active(db, PLUGIN_NAME)
    tracks = get_entity_tracks("scene", scene_id)
    cluster_ids: set[int] = set()
    for t in tracks:
        if t.label == "face" and t.cluster_id:
            cluster_ids.add(t.cluster_id)
    if not cluster_ids:
        return {"faces": []}

    clusters = get_clusters_by_ids(list(cluster_ids))
    thumb_base = "/api/v1/plugins/skier_aitagging/faces/clusters"
    result = []
    for c in clusters:
        if c.status == "merged_away":
            continue
        counts = get_entity_count_by_type(c.id)
        result.append({
            "id": c.id,
            "label": c.label,
            "status": c.status,
            "performer_id": c.performer_id,
            "sample_count": c.sample_count,
            "quality_score": c.quality_score,
            "scene_count": counts["scene_count"],
            "image_count": counts["image_count"],
            "thumbnail_url": f"{thumb_base}/{c.id}/thumbnail",
        })
    return {"faces": result}


@router.delete("/clusters/{cluster_id}/exemplars/{embedding_id}")
async def delete_cluster_exemplar(
    cluster_id: int,
    embedding_id: int,
    db: Session = Depends(get_db),
):
    """Remove a single exemplar from a cluster.

    After removal the cluster's exemplar set and centroid are recomputed.
    Non-exemplar embeddings that no longer meet the similarity threshold
    are automatically unassigned from the cluster.

    If the cluster is linked to a performer, entities that lose all face
    links to that performer (and where we originally assigned the performer)
    will have the performer removed from them in Stash.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    # Verify the embedding exists and is actually an exemplar of this cluster
    embs = get_cluster_embeddings(cluster_id, exemplars_only=True)
    target = next((e for e in embs if e.id == embedding_id), None)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail="Embedding not found or is not an exemplar of this cluster",
        )

    # Prevent removing the last exemplar — the cluster would become empty
    if len(embs) <= 1:
        raise HTTPException(
            status_code=400,
            detail="Cannot remove the last exemplar from a cluster",
        )

    performer_id = cluster.performer_id

    # Snapshot entity pairs BEFORE removal (for performer cascade)
    entities_before = set(get_cluster_entity_pairs(cluster_id)) if performer_id else set()

    _log.debug(
        "DELETE exemplar: cluster=%d embedding=%d, exemplars_before=%d ids=%s",
        cluster_id, embedding_id, len(embs), [e.id for e in embs],
    )

    try:
        result = remove_exemplar_from_cluster(embedding_id, cluster_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Verify the removal actually took effect
    embs_after = get_cluster_embeddings(cluster_id, exemplars_only=True)
    _log.debug(
        "DELETE exemplar: cluster=%d, result=%s, exemplars_after=%d ids=%s",
        cluster_id, result, len(embs_after), [e.id for e in embs_after],
    )

    # Invalidate thumbnail cache for this cluster
    _invalidate_cluster_thumbnails(cluster_id)

    # Cascade performer unassignment for entities that lost all face links
    performers_removed = 0
    if performer_id and entities_before:
        performers_removed = _cascade_performer_unassignment(
            cluster_id, performer_id, entities_before,
        )

    # Build fresh cluster detail so the frontend can update immediately
    # without needing a separate GET request.
    updated_detail = await _build_cluster_detail(cluster_id)

    return {
        "status": "ok",
        "removed_entities": result["removed_entities"],
        "remaining_exemplars": result["remaining_exemplars"],
        "performers_removed": performers_removed,
        "cluster": updated_detail,
    }


def _cascade_performer_unassignment(
    cluster_id: int,
    performer_id: int,
    entities_before: set[tuple[str, int]],
) -> int:
    """Remove a performer from entities that lost all face links after pruning.

    Only removes the performer from entities where:
      1. The entity no longer has any embedding in this cluster, AND
      2. No other cluster's assignment record links this performer to
         the entity (checked via ``face_performer_assignments``), AND
      3. We were the ones who originally assigned the performer (we have
         an assignment record for it — pre-existing performers are never
         recorded).

    Returns the number of entities from which the performer was removed.
    """
    # Snapshot entity pairs AFTER removal
    entities_after = set(get_cluster_entity_pairs(cluster_id))
    lost_pairs = list(entities_before - entities_after)
    if not lost_pairs:
        return 0

    # Delete our assignment records for the lost entity+cluster pairs
    delete_performer_assignments_for_cluster(cluster_id, performer_id, lost_pairs)

    # Check which lost entities are truly orphaned (no remaining assignment
    # records from ANY cluster for this performer)
    orphaned = get_orphaned_performer_entities(performer_id, lost_pairs)
    if not orphaned:
        return 0

    # Remove the performer from orphaned entities in Stash
    orphaned_scenes = [eid for etype, eid in orphaned if etype == "scene"]
    orphaned_images = [eid for etype, eid in orphaned if etype == "image"]

    if orphaned_scenes:
        try:
            stash_api.remove_performer_from_scenes(orphaned_scenes, performer_id)
        except Exception:
            _log.exception(
                "Failed to remove performer %s from %d orphaned scene(s)",
                performer_id, len(orphaned_scenes),
            )

    if orphaned_images:
        try:
            stash_api.remove_performer_from_images(orphaned_images, performer_id)
        except Exception:
            _log.exception(
                "Failed to remove performer %s from %d orphaned image(s)",
                performer_id, len(orphaned_images),
            )

    total = len(orphaned)
    if total:
        _log.info(
            "Removed performer %s from %d entity(ies) after pruning cluster %s "
            "(scenes=%d, images=%d)",
            performer_id, total, cluster_id,
            len(orphaned_scenes), len(orphaned_images),
        )
    return total


def _invalidate_cluster_thumbnails(cluster_id: int) -> None:
    """Delete cached thumbnail files for a cluster so they are regenerated."""
    try:
        cache_dir = _get_thumbnail_cache_dir()
        if not cache_dir.is_dir():
            return
        prefix = f"{cluster_id}_"
        for p in cache_dir.iterdir():
            if p.name.startswith(prefix) and p.is_file():
                p.unlink(missing_ok=True)
    except Exception:
        _log.debug("Failed to invalidate thumbnails for cluster %d", cluster_id, exc_info=True)


@router.post("/clusters/{cluster_id}/link")
async def link_cluster_to_performer(
    cluster_id: int,
    payload: LinkPerformerRequest,
    db: Session = Depends(get_db),
):
    """Link a face cluster to a Stash performer.

    After updating the cluster record, retroactively applies the performer
    to every Stash scene and image where the cluster's embeddings appear.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    link_performer(cluster_id, payload.performer_id, label=_resolve_performer_name(payload))

    matched_ref = _get_ref_by_id(cluster.stashdb_match_id) if getattr(cluster, "stashdb_match_id", None) else None
    if matched_ref is not None:
        await _set_local_performer_id_async(matched_ref.id, payload.performer_id)
        if payload.hydrate_from_stashdb:
            try:
                await _hydrate_performer_from_stashdb_ref(payload.performer_id, matched_ref)
            except Exception:
                _log.exception(
                    "Failed to hydrate performer %s from StashDB ref %s",
                    payload.performer_id, matched_ref.id,
                )

    # Retroactively apply the performer to all scenes/images for this cluster
    applied = _apply_performer_to_cluster_entities(cluster_id, payload.performer_id)

    # Optionally set the performer's image from the face crop
    if payload.set_performer_image:
        try:
            await _set_performer_image_from_cluster(
                cluster_id,
                payload.performer_id,
                thumbnail_index=payload.thumbnail_index,
            )
        except Exception:
            _log.exception(
                "Failed to set performer %s image from cluster %d",
                payload.performer_id, cluster_id,
            )

    return {"status": "ok", **applied}


class RejectSuggestionRequest(BaseModel):
    performer_id: int


@router.post("/clusters/{cluster_id}/reject-stashdb")
async def reject_stashdb_suggestion(cluster_id: int, db: Session = Depends(get_db)):
    """Mark the cluster's StashDB match as wrong.

    The match record (stashdb_match_id / stashdb_match_score) is preserved
    for reference, but the cluster will be excluded from batch auto-link and
    bulk operations that use the suggestion.  The top_suggestion badge on the
    card will show a rejected state with an undo button.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    with get_session_local()() as sess:
        sess.execute(
            sa.update(FaceCluster)
            .where(FaceCluster.id == cluster_id)
            .values(stashdb_suggestion_rejected=True)
        )
        sess.commit()
    return {"status": "ok"}


@router.delete("/clusters/{cluster_id}/reject-stashdb")
async def undo_reject_stashdb_suggestion(cluster_id: int, db: Session = Depends(get_db)):
    """Remove the StashDB rejection flag from a cluster."""
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    with get_session_local()() as sess:
        sess.execute(
            sa.update(FaceCluster)
            .where(FaceCluster.id == cluster_id)
            .values(stashdb_suggestion_rejected=False)
        )
        sess.commit()
    return {"status": "ok"}


@router.post("/clusters/{cluster_id}/reject-suggestion")
async def reject_co_occurrence_suggestion(
    cluster_id: int,
    payload: RejectSuggestionRequest,
    db: Session = Depends(get_db),
):
    """Mark a co-occurrence suggested performer as wrong for this cluster.

    The performer ID is appended to the cluster's rejected list.  It will be
    excluded from future suggestions and batch operations for this cluster.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    import json as _json
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    with get_session_local()() as sess:
        row = sess.execute(
            sa.select(FaceCluster.rejected_performer_ids)
            .where(FaceCluster.id == cluster_id)
        ).scalar_one_or_none()
        existing: list[int] = []
        if row:
            try:
                existing = _json.loads(row)
            except Exception:
                pass
        if payload.performer_id not in existing:
            existing.append(payload.performer_id)
        sess.execute(
            sa.update(FaceCluster)
            .where(FaceCluster.id == cluster_id)
            .values(rejected_performer_ids=_json.dumps(existing))
        )
        sess.commit()
    return {"status": "ok", "rejected_performer_ids": existing}


@router.post("/clusters/{cluster_id}/unlink")
async def unlink_face_cluster(cluster_id: int, db: Session = Depends(get_db)):
    """Remove performer link from a face cluster."""
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    if not getattr(cluster, "performer_id", None):
        raise HTTPException(status_code=400, detail="Cluster is not linked to a performer")

    from stash_ai_server.db.detection_store import unlink_performer_async
    await unlink_performer_async(cluster_id)
    return {"status": "ok"}


@router.delete("/clusters/{cluster_id}")
async def delete_face_cluster(cluster_id: int, db: Session = Depends(get_db)):
    """Permanently delete a face cluster and all associated data.

    Removes embeddings, unlinks detection tracks, deletes performer
    assignments, deletes cached face crop thumbnails, and removes the
    cluster row.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    _invalidate_cluster_thumbnails(cluster_id)
    await delete_cluster_async(cluster_id)
    return {"status": "ok"}


class BulkDeleteRequest(BaseModel):
    cluster_ids: list[int]


@router.post("/clusters/bulk-delete")
async def bulk_delete_face_clusters(
    payload: BulkDeleteRequest,
    db: Session = Depends(get_db),
):
    """Permanently delete multiple face clusters at once.

    Useful for cleaning up low-appearance clusters after a large scan.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    if not payload.cluster_ids:
        return {"status": "ok", "deleted": 0}
    # Delete thumbnails for each cluster
    for cid in payload.cluster_ids:
        _invalidate_cluster_thumbnails(cid)
    deleted = delete_clusters_bulk(payload.cluster_ids)
    return {"status": "ok", "deleted": deleted}


class CreatePerformerRequest(BaseModel):
    name: str = "Unidentified"
    set_performer_image: bool = True
    thumbnail_index: int | None = None  # which exemplar index to use for image


@router.post("/clusters/{cluster_id}/create-performer")
async def create_performer_for_cluster(
    cluster_id: int,
    payload: CreatePerformerRequest,
    db: Session = Depends(get_db),
):
    """Create a new performer in Stash and link the cluster to it.

    Useful when the cluster shows a person not yet in the Stash database.
    A default name of ``Unidentified #<cluster_id>`` is used when no name
    is supplied.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    name = payload.name.strip() if payload.name else f"Unidentified #{cluster_id}"
    if not name or name == "Unidentified":
        name = f"Unidentified #{cluster_id}"

    performer_id = stash_api.create_performer(name)
    if performer_id is None:
        # Create failed (likely "already exists") — find existing performer by name
        existing = _find_local_performers_by_name(name)
        if existing:
            performer_id = existing[0]["id"]
            _log.info(
                "create_performer_for_cluster: using existing performer '%s' (id=%s) for cluster %d",
                existing[0]["name"], performer_id, cluster_id,
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create performer in Stash")

    link_performer(cluster_id, performer_id, label=name)

    # Retroactively apply performer to scenes/images
    applied = _apply_performer_to_cluster_entities(cluster_id, performer_id)

    # Optionally set performer image from face crop
    if payload.set_performer_image:
        try:
            await _set_performer_image_from_cluster(
                cluster_id, performer_id, thumbnail_index=payload.thumbnail_index,
            )
        except Exception:
            _log.exception(
                "Failed to set performer %s image from cluster %d",
                performer_id, cluster_id,
            )

    return {"status": "ok", "performer_id": performer_id, "performer_name": name, **applied}


@router.post("/clusters/merge")
async def merge_face_clusters(
    payload: MergeClustersRequest,
    db: Session = Depends(get_db),
):
    """Merge one cluster into another."""
    _require_plugin_active(db, PLUGIN_NAME)
    surviving = get_cluster_by_id(payload.surviving_id)
    absorbed = get_cluster_by_id(payload.absorbed_id)
    if surviving is None or absorbed is None:
        raise HTTPException(status_code=404, detail="One or both clusters not found")

    merge_clusters(payload.surviving_id, payload.absorbed_id)
    return {"status": "ok"}


class BulkLinkToSuggestedRequest(BaseModel):
    cluster_ids: list[int]


@router.post("/clusters/bulk-link-suggested")
async def bulk_link_clusters_to_suggested(
    payload: BulkLinkToSuggestedRequest,
    db: Session = Depends(get_db),
):
    """Link multiple clusters to their top-suggested performer in one shot.

    For each cluster:
    - Skips if already linked, ignored, or if suggestions are rejected.
    - If the suggestion has a local ``performer_id``, links directly.
    - If the suggestion is StashDB-only (``confidence == "stashdb"``),
      creates a new performer from the StashDB ref, hydrates it, then links.
    - Retroactively applies performers to scenes/images.

    Returns a summary of how many were linked, skipped, and any errors.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    if not payload.cluster_ids:
        return {"status": "ok", "linked": 0, "skipped": 0, "errors": 0, "details": []}

    suggestions = _compute_batch_top_suggestions(payload.cluster_ids)

    linked = 0
    skipped = 0
    errors = 0
    details: list[dict] = []

    for cid in payload.cluster_ids:
        cluster = get_cluster_by_id(cid)
        if cluster is None:
            skipped += 1
            details.append({"cluster_id": cid, "status": "skipped", "reason": "not found"})
            continue
        if cluster.status != "unidentified":
            skipped += 1
            details.append({"cluster_id": cid, "status": "skipped", "reason": cluster.status})
            continue

        suggestion = suggestions.get(cid)
        if suggestion is None:
            skipped += 1
            details.append({"cluster_id": cid, "status": "skipped", "reason": "no suggestion"})
            continue

        performer_id = suggestion.get("performer_id")
        performer_name = suggestion.get("performer_name", "")

        # StashDB-only: need to create the performer first
        if performer_id is None and suggestion.get("confidence") == "stashdb":
            matched_ref = (
                _get_ref_by_id(cluster.stashdb_match_id)
                if getattr(cluster, "stashdb_match_id", None)
                else None
            )
            if matched_ref is None:
                skipped += 1
                details.append({"cluster_id": cid, "status": "skipped", "reason": "no StashDB ref"})
                continue

            endpoint = getattr(matched_ref, "source_endpoint", None) or "https://stashdb.org/graphql"
            stashdb_id = getattr(matched_ref, "stashdb_id", None)
            created_new = False
            performer_id = stash_api.create_performer(
                performer_name or f"Unidentified #{cid}",
                stash_ids=[{"stash_id": stashdb_id, "endpoint": endpoint}] if stashdb_id else None,
                disambiguation=getattr(matched_ref, "disambiguation", None),
            )
            if performer_id is not None:
                created_new = True
            else:
                # Create failed (likely "already exists") — find existing performer by name
                _log.info(
                    "Bulk-link: create_performer failed for '%s', looking up existing performer",
                    performer_name,
                )
                existing = _find_local_performers_by_name(performer_name or "")
                if existing:
                    performer_id = existing[0]["id"]
                    _log.info(
                        "Bulk-link: found existing performer '%s' (id=%s) for cluster %d",
                        existing[0]["name"], performer_id, cid,
                    )
                    # Attach stash_ids to the existing performer if not already set
                    if stashdb_id:
                        try:
                            stash_api.update_performer(
                                performer_id,
                                stash_ids=[{"stash_id": stashdb_id, "endpoint": endpoint}],
                            )
                        except Exception:
                            _log.debug("Bulk-link: failed to set stash_ids on existing performer %s", performer_id)
                else:
                    _log.warning(
                        "Bulk-link: no existing performer found for name '%s' (cluster %d)",
                        performer_name, cid,
                    )

            if performer_id is None:
                errors += 1
                details.append({"cluster_id": cid, "status": "error", "reason": "failed to create performer"})
                continue

            # Set local_performer_id on the ref
            try:
                await _set_local_performer_id_async(matched_ref.id, performer_id)
            except Exception:
                pass

            # Hydrate (always — even for existing performers that may lack data)
            try:
                await _hydrate_performer_from_stashdb_ref(performer_id, matched_ref)
            except Exception:
                _log.debug("Bulk-link: hydration failed for cluster %d", cid, exc_info=True)

        if performer_id is None:
            skipped += 1
            details.append({"cluster_id": cid, "status": "skipped", "reason": "no performer_id"})
            continue

        try:
            rows = link_performer(cid, performer_id, label=performer_name or None)
            if rows == 0:
                _log.error(
                    "Bulk-link: link_performer updated 0 rows for cluster %d (performer %s). "
                    "Attempting direct fallback update.",
                    cid, performer_id,
                )
                # Fallback: direct update in a fresh session
                with get_session_local()() as _sess:
                    _sess.execute(
                        sa.update(FaceCluster)
                        .where(FaceCluster.id == cid)
                        .values(
                            performer_id=performer_id,
                            status="identified",
                            label=performer_name or None,
                        )
                    )
                    _sess.commit()

            # Verify the link actually persisted
            verify = get_cluster_by_id(cid)
            if verify is None or verify.status != "identified":
                _log.error(
                    "Bulk-link: POST-LINK VERIFICATION FAILED for cluster %d. "
                    "Status is '%s' (expected 'identified'). rows=%d, performer_id=%s",
                    cid, getattr(verify, 'status', 'NOT FOUND'), rows, performer_id,
                )
                errors += 1
                details.append({"cluster_id": cid, "status": "error", "reason": "verification failed"})
                continue

            _apply_performer_to_cluster_entities(cid, performer_id)
            linked += 1
            details.append({
                "cluster_id": cid,
                "status": "linked",
                "performer_id": performer_id,
                "performer_name": performer_name,
            })
        except Exception:
            _log.exception("Bulk-link: failed to link cluster %d", cid)
            errors += 1
            details.append({"cluster_id": cid, "status": "error", "reason": "link failed"})

    return {
        "status": "ok",
        "linked": linked,
        "skipped": skipped,
        "errors": errors,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Thumbnail endpoint
# ---------------------------------------------------------------------------

def _get_best_exemplar_data(cluster_id: int) -> tuple[str, int, list[float], float | None] | None:
    """Return (entity_type, entity_id, bbox, timestamp_s) for the best exemplar.

    Uses a plain column-level query so the result is detach-safe (no ORM lazy
    loading issues with ARRAY columns after session close).
    """
    candidates = _get_exemplar_candidates(cluster_id, limit=1)
    return candidates[0] if candidates else None


def _get_exemplar_candidates(
    cluster_id: int,
    limit: int = 5,
) -> list[tuple[str, int, list[float], float | None]]:
    """Return up to *limit* exemplar candidates ordered by detection score.

    Each entry is ``(entity_type, entity_id, bbox, timestamp_s)``.
    """
    from stash_ai_server.models.detections import FaceEmbedding as FE
    from stash_ai_server.db.session import get_session_local
    from sqlalchemy import select as sa_select

    try:
        session_factory = get_session_local()
        with session_factory() as session:
            stmt = (
                sa_select(FE.entity_type, FE.entity_id, FE.bbox, FE.timestamp_s)
                .where(FE.cluster_id == cluster_id, FE.is_exemplar.is_(True))
                .order_by(FE.score.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).all()
            results: list[tuple[str, int, list[float], float | None]] = []
            for row in rows:
                bbox_raw = row[2]
                bbox = list(bbox_raw) if bbox_raw else None
                if not bbox or len(bbox) != 4:
                    continue
                results.append((
                    row[0],
                    int(row[1]),
                    bbox,
                    float(row[3]) if row[3] is not None else None,
                ))
            return results
    except Exception as exc:
        _log.exception("_get_exemplar_candidates failed for cluster %d: %s", cluster_id, exc)
        return []


def _get_scene_file_path(scene_id: int) -> str | None:
    """Return the local filesystem path for a scene's video file.

    Queries Stash for the scene path and applies backend path mappings.
    """
    from stash_ai_server.utils.path_mutation import mutate_path_for_backend

    try:
        path, _tags, _dur = stash_api.get_scene_path_and_tags_and_duration(scene_id)
    except Exception as exc:
        _log.warning("Failed to get scene path for scene %d: %s", scene_id, exc)
        return None
    if not path:
        return None
    return mutate_path_for_backend(path)


def _extract_video_frame(video_path: str, timestamp_s: float | None) -> bytes | None:
    """Extract a single frame from a video file using OpenCV.

    Returns the frame as PNG bytes, or ``None`` on failure.
    If ``timestamp_s`` exceeds the video duration the seek is clamped to
    a few seconds before the end so a valid frame is still returned.
    """
    try:
        import cv2
    except ImportError:
        _log.warning("opencv-python-headless is not installed; cannot extract video frames")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _log.warning("Could not open video: %s", video_path)
        return None

    try:
        if timestamp_s is not None and timestamp_s > 0:
            # Clamp to video duration so seeking beyond the end doesn't
            # silently fail to read a frame.
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            duration_s = frame_count / fps if fps > 0 else 0
            if duration_s > 0 and timestamp_s > duration_s - 1.0:
                timestamp_s = max(0, duration_s - 2.0)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)

        ok, frame = cap.read()
        if not ok or frame is None:
            _log.warning("Failed to read frame at %.2fs from %s", timestamp_s or 0, video_path)
            return None

        # Encode as PNG for lossless transfer to PIL
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            return None
        return buf.tobytes()
    finally:
        cap.release()


def _face_sharpness(
    source_bytes: bytes,
    bbox: list[float],
) -> float:
    """Return a sharpness score for the face region of an image.

    Uses the variance of the Laplacian on the face crop.  Higher = sharper.
    Returns ``0.0`` on any error.
    """
    try:
        import cv2
        import numpy as np

        arr = np.frombuffer(source_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        h, w = img.shape[:2]
        x1 = max(0, int(bbox[0] * w))
        y1 = max(0, int(bbox[1] * h))
        x2 = min(w, int(bbox[2] * w))
        y2 = min(h, int(bbox[3] * h))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop = img[y1:y2, x1:x2]
        return float(cv2.Laplacian(crop, cv2.CV_64F).var())
    except Exception:
        return 0.0


async def _fetch_source_via_stash(entity_type: str, entity_id: int) -> bytes | None:
    """Fetch a source image/frame via the Stash HTTP API."""
    base_url = stash_api._effective_url or stash_api.stash_url
    if not base_url:
        return None
    if entity_type == "scene":
        img_url = f"{base_url}/scene/{entity_id}/screenshot"
    elif entity_type == "image":
        img_url = f"{base_url}/image/{entity_id}/image"
    else:
        return None
    headers = {}
    if stash_api.api_key:
        headers["ApiKey"] = stash_api.api_key
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(img_url, headers=headers, timeout=15.0, follow_redirects=True)
            if resp.status_code == 200:
                return resp.content
    except httpx.HTTPError:
        pass
    return None


@router.get("/clusters/{cluster_id}/thumbnail-count")
async def get_cluster_thumbnail_count(
    cluster_id: int,
    db: Session = Depends(get_db),
):
    """Return the number of exemplar images available for cycling."""
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    candidates = _get_exemplar_candidates(cluster_id, limit=20)
    return {"count": len(candidates)}


@router.get("/clusters/{cluster_id}/thumbnail")
async def get_cluster_thumbnail(
    cluster_id: int,
    size: int = Query(150, ge=32, le=512),
    index: int = Query(0, ge=0, description="Exemplar index (0 = sharpest)"),
    pad: float = Query(0.2, ge=0.0, le=1.0, description="Padding ratio around face bbox"),
    db: Session = Depends(get_db),
):
    """Return a face crop thumbnail for a cluster exemplar.

    On the first request for any index of a given cluster the *all*
    indices are generated and cached on disk at a canonical size
    (``_CANONICAL_THUMB_SIZE``).  Subsequent requests — regardless of the
    requested ``size`` — are served from that single cache entry.
    """
    _require_plugin_active(db, PLUGIN_NAME)

    # Normalise to the canonical size so every UI context (grid, detail,
    # review panel, etc.) shares one set of cached thumbnails.
    size = _CANONICAL_THUMB_SIZE

    cache_dir = _get_thumbnail_cache_dir()
    pad_key = int(pad * 100)
    cache_path = cache_dir / f"{cluster_id}_{size}_i{index}_p{pad_key}.jpg"

    # Serve from cache if available
    if cache_path.exists():
        return Response(
            content=cache_path.read_bytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    # Get up to 20 exemplar candidates ordered by detection score
    candidates = _get_exemplar_candidates(cluster_id, limit=20)
    if not candidates:
        raise HTTPException(status_code=404, detail="No exemplar found")

    # ------------------------------------------------------------------
    # Rank ALL candidates by sharpness so we can batch-generate every
    # index variant in a single pass (the expensive part is fetching
    # source images — might as well crop them all now).
    # ------------------------------------------------------------------
    scored: list[tuple[float, int, bytes, list[float]]] = []

    video_path: str | None = None
    first_entity_type = candidates[0][0]
    first_entity_id = candidates[0][1]

    if first_entity_type == "scene":
        video_path = await asyncio.to_thread(_get_scene_file_path, first_entity_id)
        if video_path and not Path(video_path).is_file():
            video_path = None

    for cand_idx, (entity_type, entity_id, bbox, timestamp_s) in enumerate(candidates):
        if not bbox or len(bbox) != 4:
            continue

        source_bytes: bytes | None = None

        if entity_type == "scene" and video_path:
            source_bytes = await asyncio.to_thread(
                _extract_video_frame, video_path, timestamp_s,
            )

        if source_bytes is None:
            source_bytes = await _fetch_source_via_stash(entity_type, entity_id)

        if source_bytes is None:
            continue

        sharpness = await asyncio.to_thread(_face_sharpness, source_bytes, bbox)
        scored.append((sharpness, cand_idx, source_bytes, bbox))

    if not scored:
        raise HTTPException(status_code=404, detail="Could not obtain source image for any exemplar")

    scored.sort(key=lambda t: -t[0])

    # ------------------------------------------------------------------
    # Crop & cache EVERY index at once, then return the requested one.
    # ------------------------------------------------------------------
    requested_bytes: bytes | None = None

    try:
        from PIL import Image

        for rank, (_, _, src_bytes, bbox) in enumerate(scored):
            this_cache = cache_dir / f"{cluster_id}_{size}_i{rank}_p{pad_key}.jpg"
            if this_cache.exists():
                if rank == index:
                    requested_bytes = this_cache.read_bytes()
                continue

            img = Image.open(io.BytesIO(src_bytes))
            w, h = img.size

            clamped = [max(0.0, min(1.0, v)) for v in bbox]
            x1, y1, x2, y2 = clamped
            if all(0 <= v <= 1.0 for v in clamped):
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            else:
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            pad_w = int((x2 - x1) * pad)
            pad_h = int((y2 - y1) * pad)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            face_crop = img.crop((x1, y1, x2, y2))
            if face_crop.mode not in ("RGB", "L"):
                face_crop = face_crop.convert("RGB")

            crop_w, crop_h = face_crop.size
            if crop_w >= crop_h:
                new_w = size
                new_h = max(1, int(size * crop_h / crop_w))
            else:
                new_h = size
                new_w = max(1, int(size * crop_w / crop_h))
            face_crop = face_crop.resize((new_w, new_h), Image.LANCZOS)

            buf = io.BytesIO()
            face_crop.save(buf, format="JPEG", quality=85)
            jpeg_bytes = buf.getvalue()

            try:
                this_cache.write_bytes(jpeg_bytes)
            except Exception:
                _log.warning("Failed to cache thumbnail for cluster %d index %d at %s", cluster_id, rank, this_cache, exc_info=True)

            if rank == index:
                requested_bytes = jpeg_bytes

        # If the requested index exceeds available, fall back to last
        if requested_bytes is None:
            fallback_idx = min(index, len(scored) - 1)
            fallback_cache = cache_dir / f"{cluster_id}_{size}_i{fallback_idx}_p{pad_key}.jpg"
            if fallback_cache.exists():
                requested_bytes = fallback_cache.read_bytes()
            elif scored:
                # Generate on the fly as last resort
                _, _, src_bytes, bbox = scored[fallback_idx]
                img = Image.open(io.BytesIO(src_bytes))
                w, h = img.size
                clamped = [max(0.0, min(1.0, v)) for v in bbox]
                x1, y1, x2, y2 = clamped
                if all(0 <= v <= 1.0 for v in clamped):
                    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                else:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                pad_w = int((x2 - x1) * pad)
                pad_h = int((y2 - y1) * pad)
                x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
                face_crop = img.crop((x1, y1, x2, y2))
                if face_crop.mode not in ("RGB", "L"):
                    face_crop = face_crop.convert("RGB")
                crop_w, crop_h = face_crop.size
                if crop_w >= crop_h:
                    new_w, new_h = size, max(1, int(size * crop_h / crop_w))
                else:
                    new_h, new_w = size, max(1, int(size * crop_w / crop_h))
                face_crop = face_crop.resize((new_w, new_h), Image.LANCZOS)
                buf = io.BytesIO()
                face_crop.save(buf, format="JPEG", quality=85)
                requested_bytes = buf.getvalue()

        if requested_bytes is None:
            raise HTTPException(status_code=404, detail="Face bounding box too small")

        return Response(
            content=requested_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    except ImportError:
        raise HTTPException(status_code=500, detail="PIL/Pillow not available for thumbnail generation")
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception(
            "Failed to generate thumbnail for cluster %d: %s",
            cluster_id, exc,
        )
        raise HTTPException(status_code=500, detail=f"Thumbnail generation failed: {exc}")


async def precompute_cluster_thumbnails(cluster_ids: list[int]) -> int:
    """Pre-generate cached thumbnail crops for the given cluster IDs.

    Called after scene/image processing so the FacesHub grid loads instantly.
    Returns the number of clusters for which at least one thumbnail was cached.
    """
    if not cluster_ids:
        return 0

    size = _CANONICAL_THUMB_SIZE
    pad = 0.2
    pad_key = int(pad * 100)
    cache_dir = _get_thumbnail_cache_dir()
    cached_count = 0

    try:
        from PIL import Image
    except ImportError:
        _log.debug("PIL not available; skipping thumbnail precompute")
        return 0

    for cluster_id in cluster_ids:
        # Skip if index-0 thumbnail already cached
        idx0_path = cache_dir / f"{cluster_id}_{size}_i0_p{pad_key}.jpg"
        if idx0_path.exists():
            continue

        try:
            candidates = _get_exemplar_candidates(cluster_id, limit=20)
            if not candidates:
                continue

            # Resolve video path for scene-based candidates
            video_path: str | None = None
            first_et, first_eid = candidates[0][0], candidates[0][1]
            if first_et == "scene":
                video_path = await asyncio.to_thread(_get_scene_file_path, first_eid)
                if video_path and not Path(video_path).is_file():
                    video_path = None

            scored: list[tuple[float, int, bytes, list[float]]] = []
            for cand_idx, (entity_type, entity_id, bbox, timestamp_s) in enumerate(candidates):
                if not bbox or len(bbox) != 4:
                    continue
                source_bytes: bytes | None = None
                if entity_type == "scene" and video_path:
                    source_bytes = await asyncio.to_thread(
                        _extract_video_frame, video_path, timestamp_s,
                    )
                if source_bytes is None:
                    source_bytes = await _fetch_source_via_stash(entity_type, entity_id)
                if source_bytes is None:
                    continue
                sharpness = await asyncio.to_thread(_face_sharpness, source_bytes, bbox)
                scored.append((sharpness, cand_idx, source_bytes, bbox))

            if not scored:
                continue

            scored.sort(key=lambda t: -t[0])
            any_cached = False

            for rank, (_, _, src_bytes, bbox) in enumerate(scored):
                this_cache = cache_dir / f"{cluster_id}_{size}_i{rank}_p{pad_key}.jpg"
                if this_cache.exists():
                    any_cached = True
                    continue
                img = Image.open(io.BytesIO(src_bytes))
                w, h = img.size
                clamped = [max(0.0, min(1.0, v)) for v in bbox]
                x1, y1, x2, y2 = clamped
                if all(0 <= v <= 1.0 for v in clamped):
                    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                else:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                pad_w = int((x2 - x1) * pad)
                pad_h = int((y2 - y1) * pad)
                x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                face_crop = img.crop((x1, y1, x2, y2))
                if face_crop.mode not in ("RGB", "L"):
                    face_crop = face_crop.convert("RGB")
                crop_w, crop_h = face_crop.size
                if crop_w >= crop_h:
                    new_w, new_h = size, max(1, int(size * crop_h / crop_w))
                else:
                    new_h, new_w = size, max(1, int(size * crop_w / crop_h))
                face_crop = face_crop.resize((new_w, new_h), Image.LANCZOS)
                buf = io.BytesIO()
                face_crop.save(buf, format="JPEG", quality=85)
                try:
                    this_cache.write_bytes(buf.getvalue())
                    any_cached = True
                except Exception:
                    _log.warning("precompute: failed to write cache for cluster %d rank %d", cluster_id, rank)

            if any_cached:
                cached_count += 1
        except Exception:
            _log.debug("precompute_cluster_thumbnails: failed for cluster %d", cluster_id, exc_info=True)

    if cached_count:
        _log.info("Pre-computed thumbnails for %d/%d cluster(s)", cached_count, len(cluster_ids))
    return cached_count


# ---------------------------------------------------------------------------
# Suggested performers (cross-DB co-occurrence)
# ---------------------------------------------------------------------------

def _pick_column(table: sa.Table | None, *names: str):
    """Pick the first column matching one of the given names."""
    if table is None:
        return None
    for n in names:
        c = table.c.get(n)
        if c is not None:
            return c
    return None


def _label_or_literal(column, alias: str, default: Any = None):
    """Label a column or return a literal if the column is None."""
    if column is None:
        return sa.literal(default).label(alias)
    try:
        return column.label(alias)
    except Exception:
        return sa.literal(default).label(alias)


def _query_stash_co_occurrence(
    scene_ids: list[int],
    image_ids: list[int],
) -> dict[int, dict]:
    """Query Stash SQLite DB for performers co-occurring with given scenes/images.

    Returns: { performer_id: { name, image, scene_count, image_count,
                 co_occurrence_count, solo_scene_count, solo_image_count } }
    """
    performers_table = stash_db.get_stash_table("performers", required=False)
    if performers_table is None:
        _log.debug("performers table not found in Stash DB; cannot compute co-occurrence")
        return {}

    perf_id_col = performers_table.c.get("id")
    perf_name_col = performers_table.c.get("name")
    perf_image_col = _pick_column(performers_table, "image_path", "image")

    if perf_id_col is None or perf_name_col is None:
        return {}

    result: dict[int, dict] = {}
    session_factory = stash_db.get_stash_sessionmaker()
    if session_factory is None:
        _log.debug("Stash DB session factory not available")
        return {}

    try:
        with session_factory() as session:
            # --- Scene co-occurrence (single query) ---
            if scene_ids:
                scene_link = stash_db.get_first_available_table(
                    "performers_scenes", "scene_performers", "performer_scenes",
                    required_columns=("scene_id", "performer_id"),
                )
                if scene_link is not None:
                    s_scene_col = scene_link.c.get("scene_id")
                    s_perf_col = scene_link.c.get("performer_id")

                    if s_scene_col is not None and s_perf_col is not None:
                        stmt = (
                            sa.select(
                                s_scene_col.label("entity_id"),
                                perf_id_col.label("performer_id"),
                                perf_name_col.label("name"),
                                _label_or_literal(perf_image_col, "image_path"),
                            )
                            .select_from(
                                scene_link.join(performers_table, perf_id_col == s_perf_col)
                            )
                            .where(s_scene_col.in_(scene_ids))
                        )

                        entity_performers: dict[int, list[int]] = defaultdict(list)
                        for row in session.execute(stmt).all():
                            try:
                                eid = int(row.entity_id)
                                pid = int(row.performer_id)
                            except (TypeError, ValueError):
                                continue
                            entity_performers[eid].append(pid)

                            if pid not in result:
                                result[pid] = {
                                    "name": row.name,
                                    "image": row.image_path or "",
                                    "scene_count": 0,
                                    "image_count": 0,
                                    "co_occurrence_count": 0,
                                    "solo_scene_count": 0,
                                    "solo_image_count": 0,
                                }
                            result[pid]["scene_count"] += 1

                        # Detect solo performers per scene
                        for eid, pids in entity_performers.items():
                            if len(pids) == 1:
                                result[pids[0]]["solo_scene_count"] += 1

            # --- Image co-occurrence (single query) ---
            if image_ids:
                image_link = stash_db.get_first_available_table(
                    "performers_images", "image_performers", "performer_images",
                    required_columns=("image_id", "performer_id"),
                )
                if image_link is not None:
                    i_image_col = image_link.c.get("image_id")
                    i_perf_col = image_link.c.get("performer_id")

                    if i_image_col is not None and i_perf_col is not None:
                        stmt = (
                            sa.select(
                                i_image_col.label("entity_id"),
                                perf_id_col.label("performer_id"),
                                perf_name_col.label("name"),
                                _label_or_literal(perf_image_col, "image_path"),
                            )
                            .select_from(
                                image_link.join(performers_table, perf_id_col == i_perf_col)
                            )
                            .where(i_image_col.in_(image_ids))
                        )

                        entity_performers_img: dict[int, list[int]] = defaultdict(list)
                        for row in session.execute(stmt).all():
                            try:
                                eid = int(row.entity_id)
                                pid = int(row.performer_id)
                            except (TypeError, ValueError):
                                continue
                            entity_performers_img[eid].append(pid)

                            if pid not in result:
                                result[pid] = {
                                    "name": row.name,
                                    "image": row.image_path or "",
                                    "scene_count": 0,
                                    "image_count": 0,
                                    "co_occurrence_count": 0,
                                    "solo_scene_count": 0,
                                    "solo_image_count": 0,
                                }
                            result[pid]["image_count"] += 1

                        for eid, pids in entity_performers_img.items():
                            if len(pids) == 1:
                                result[pids[0]]["solo_image_count"] += 1

    except Exception:
        _log.exception("Error querying Stash DB for co-occurrence")
        return {}

    # Compute combined counts
    for data in result.values():
        data["co_occurrence_count"] = data["scene_count"] + data["image_count"]

    return result


# ---------------------------------------------------------------------------
# Batch top-suggestion helpers (for cluster grid enrichment)
# ---------------------------------------------------------------------------

def _query_entity_performer_map(
    scene_ids: list[int],
    image_ids: list[int],
) -> tuple[dict[tuple[str, int], list[int]], dict[int, dict]]:
    """Query Stash SQLite for per-entity performer lists.

    Returns
    -------
    entity_performers : dict
        ``{("scene", sid): [pid, ...], ("image", iid): [pid, ...]}``
    performer_info : dict
        ``{pid: {"name": str, "image": str}}``
    """
    entity_performers: dict[tuple[str, int], list[int]] = {}
    performer_info: dict[int, dict] = {}

    performers_table = stash_db.get_stash_table("performers", required=False)
    if performers_table is None:
        return entity_performers, performer_info

    perf_id_col = performers_table.c.get("id")
    perf_name_col = performers_table.c.get("name")
    if perf_id_col is None or perf_name_col is None:
        return entity_performers, performer_info

    session_factory = stash_db.get_stash_sessionmaker()
    if session_factory is None:
        return entity_performers, performer_info

    try:
        with session_factory() as session:
            if scene_ids:
                scene_link = stash_db.get_first_available_table(
                    "performers_scenes", "scene_performers", "performer_scenes",
                    required_columns=("scene_id", "performer_id"),
                )
                if scene_link is not None:
                    s_scene_col = scene_link.c.get("scene_id")
                    s_perf_col = scene_link.c.get("performer_id")
                    if s_scene_col is not None and s_perf_col is not None:
                        stmt = (
                            sa.select(
                                s_scene_col.label("entity_id"),
                                perf_id_col.label("performer_id"),
                                perf_name_col.label("name"),
                            )
                            .select_from(
                                scene_link.join(performers_table, perf_id_col == s_perf_col)
                            )
                            .where(s_scene_col.in_(scene_ids))
                        )
                        for row in session.execute(stmt).all():
                            try:
                                eid = int(row.entity_id)
                                pid = int(row.performer_id)
                            except (TypeError, ValueError):
                                continue
                            key = ("scene", eid)
                            entity_performers.setdefault(key, []).append(pid)
                            if pid not in performer_info:
                                performer_info[pid] = {"name": row.name}

            if image_ids:
                image_link = stash_db.get_first_available_table(
                    "performers_images", "image_performers", "performer_images",
                    required_columns=("image_id", "performer_id"),
                )
                if image_link is not None:
                    i_image_col = image_link.c.get("image_id")
                    i_perf_col = image_link.c.get("performer_id")
                    if i_image_col is not None and i_perf_col is not None:
                        stmt = (
                            sa.select(
                                i_image_col.label("entity_id"),
                                perf_id_col.label("performer_id"),
                                perf_name_col.label("name"),
                            )
                            .select_from(
                                image_link.join(performers_table, perf_id_col == i_perf_col)
                            )
                            .where(i_image_col.in_(image_ids))
                        )
                        for row in session.execute(stmt).all():
                            try:
                                eid = int(row.entity_id)
                                pid = int(row.performer_id)
                            except (TypeError, ValueError):
                                continue
                            key = ("image", eid)
                            entity_performers.setdefault(key, []).append(pid)
                            if pid not in performer_info:
                                performer_info[pid] = {"name": row.name}
    except Exception:
        _log.exception("Error querying entity performer map from Stash DB")

    return entity_performers, performer_info


def _compute_batch_top_suggestions(
    cluster_ids: list[int],
) -> dict[int, dict | None]:
    """Compute the top co-occurrence suggestion for each cluster.

    Returns ``{cluster_id: {performer_id, performer_name, confidence,
    co_occurrence_ratio, co_occurrence_count, total_entities} | None}``.
    """
    if not cluster_ids:
        return {}

    # 1. Bulk-fetch entity pairs for all clusters
    all_entity_pairs = get_bulk_cluster_entity_pairs(cluster_ids)

    # 2. Collect all unique scene/image IDs across all clusters
    all_scene_ids: set[int] = set()
    all_image_ids: set[int] = set()
    for pairs in all_entity_pairs.values():
        for etype, eid in pairs:
            if etype == "scene":
                all_scene_ids.add(eid)
            elif etype == "image":
                all_image_ids.add(eid)

    if not all_scene_ids and not all_image_ids:
        return {cid: None for cid in cluster_ids}

    # 3. Single batch query to Stash SQLite
    entity_performers, performer_info = _query_entity_performer_map(
        list(all_scene_ids), list(all_image_ids),
    )

    # 3b. Bulk-fetch StashDB match info so we can prefer the StashDB
    #     performer over co-occurrence when they disagree.
    stashdb_local_pids: dict[int, int] = {}  # cluster_id -> local_performer_id
    stashdb_ref_names: dict[int, str] = {}   # cluster_id -> StashDB performer name
    try:
        with get_session_local()() as sess:
            rows = sess.execute(
                sa.select(
                    FaceCluster.id,
                    StashDBPerformerRef.local_performer_id,
                    StashDBPerformerRef.name,
                )
                .join(StashDBPerformerRef, FaceCluster.stashdb_match_id == StashDBPerformerRef.id)
                .where(
                    FaceCluster.id.in_(cluster_ids),
                    FaceCluster.stashdb_match_id.isnot(None),
                )
            ).all()
            for r in rows:
                stashdb_ref_names[r[0]] = r[2]
                if r[1] is not None:
                    stashdb_local_pids[r[0]] = r[1]
    except Exception:
        _log.debug("Could not fetch StashDB info for batch", exc_info=True)

    # 3c. Bulk-fetch rejection state to exclude user-dismissed suggestions.
    rejected_stashdb_cids: set[int] = set()   # cluster IDs with stashdb rejected
    rejected_performers: dict[int, set[int]] = {}  # cluster_id -> set of performer IDs
    try:
        import json as _json
        with get_session_local()() as sess:
            rej_rows = sess.execute(
                sa.select(
                    FaceCluster.id,
                    FaceCluster.stashdb_suggestion_rejected,
                    FaceCluster.rejected_performer_ids,
                )
                .where(
                    FaceCluster.id.in_(cluster_ids),
                    sa.or_(
                        FaceCluster.stashdb_suggestion_rejected == sa.true(),
                        FaceCluster.rejected_performer_ids.isnot(None),
                    ),
                )
            ).all()
            for r in rej_rows:
                if r[1]:
                    rejected_stashdb_cids.add(r[0])
                if r[2]:
                    try:
                        pids = _json.loads(r[2])
                        if pids:
                            rejected_performers[r[0]] = set(pids)
                    except Exception:
                        pass
    except Exception:
        _log.debug("Could not fetch rejection state for batch", exc_info=True)

    # 4. For each cluster, compute co-occurrence and pick the top suggestion
    result: dict[int, dict | None] = {}
    for cid in cluster_ids:
        pairs = all_entity_pairs.get(cid, [])
        total = len(pairs)

        # Count how many of this cluster's entities each performer appears in
        performer_counts: Counter[int] = Counter()
        if total > 0:
            for entity_key in pairs:
                for pid in entity_performers.get(entity_key, []):
                    performer_counts[pid] += 1

        # Remove explicitly rejected performers from the co-occurrence pool.
        for _rpid in rejected_performers.get(cid, set()):
            performer_counts.pop(_rpid, None)

        # When a StashDB match exists with a local performer, always prefer
        # that performer — the embedding match is more authoritative than
        # co-occurrence (which can pick the wrong person in multi-performer
        # scenes).  Skip if the user rejected this StashDB suggestion.
        stashdb_pid = stashdb_local_pids.get(cid) if cid not in rejected_stashdb_cids else None
        stashdb_name = stashdb_ref_names.get(cid) if cid not in rejected_stashdb_cids else None

        if stashdb_pid is not None and stashdb_pid in performer_counts:
            top_pid = stashdb_pid
            top_count = performer_counts[top_pid]
            ratio = top_count / total if total > 0 else 0
            confidence = "high" if ratio > 0.7 else "possible" if ratio > 0.3 else "suggested"
            info = performer_info.get(top_pid, {})
            result[cid] = {
                "performer_id": top_pid,
                "performer_name": info.get("name", stashdb_name or f"Performer #{top_pid}"),
                "confidence": confidence,
                "co_occurrence_ratio": round(ratio, 3),
                "co_occurrence_count": top_count,
                "total_entities": total,
            }
        elif stashdb_name is not None:
            # StashDB match exists but no local performer yet — show the
            # StashDB name as the suggestion so the card doesn't mislead.
            result[cid] = {
                "performer_id": stashdb_pid,  # may be None
                "performer_name": stashdb_name,
                "confidence": "stashdb",
                "co_occurrence_ratio": 0,
                "co_occurrence_count": 0,
                "total_entities": total,
            }
        elif performer_counts:
            top_pid, top_count = performer_counts.most_common(1)[0]
            ratio = top_count / total if total > 0 else 0
            confidence = "high" if ratio > 0.7 else "possible" if ratio > 0.3 else "suggested"
            info = performer_info.get(top_pid, {})
            result[cid] = {
                "performer_id": top_pid,
                "performer_name": info.get("name", f"Performer #{top_pid}"),
                "confidence": confidence,
                "co_occurrence_ratio": round(ratio, 3),
                "co_occurrence_count": top_count,
                "total_entities": total,
            }
        else:
            result[cid] = None

    return result


@router.get("/clusters/{cluster_id}/suggested-performers")
async def get_suggested_performers(
    cluster_id: int,
    db: Session = Depends(get_db),
):
    """Return Stash performers ranked by co-occurrence with this face cluster.

    When the cluster has a StashDB match with a verified local performer that
    already carries the same StashDB stash-id, that performer is injected as a
    top-priority "stashdb" confidence suggestion. If no verified local link
    exists, same-name local performers are surfaced only as non-persistent
    fallback suggestions.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    # Step 1: Get all entities where this cluster appears (from our PostgreSQL DB)
    entity_pairs = get_cluster_entity_pairs(cluster_id)
    scene_ids = [eid for etype, eid in entity_pairs if etype == "scene"]
    image_ids = [eid for etype, eid in entity_pairs if etype == "image"]
    total_entities = len(entity_pairs)

    # Step 2: Inject StashDB-based suggestions BEFORE co-occurrence
    stashdb_suggestions: list[dict] = []
    stashdb_performer_ids: set[int] = set()  # avoid duplicating in co-occurrence results
    matched_ref = None
    if getattr(cluster, "stashdb_match_id", None):
        matched_ref = _get_ref_by_id(cluster.stashdb_match_id)
    if matched_ref is not None:
        verified_local = await _get_verified_local_performer_for_ref(matched_ref, repair=True)

        if verified_local is not None:
            local_pid = verified_local["id"]
            stashdb_performer_ids.add(local_pid)
            perf_name = verified_local["name"]

            linked_cluster = get_cluster_for_performer(local_pid)
            already_linked = linked_cluster is not None and linked_cluster.id != cluster_id

            stashdb_suggestions.append({
                "performer_id": local_pid,
                "performer_name": perf_name,
                "performer_image": f"/performer/{local_pid}/image",
                "stashdb_image_url": getattr(matched_ref, "image_url", None),
                "scene_count": 0,
                "image_count": 0,
                "co_occurrence_count": 0,
                "confidence": "stashdb",
                "solo_scene_count": 0,
                "solo_image_count": 0,
                "already_linked": already_linked,
                "stashdb_match": {
                    "stashdb_id": matched_ref.stashdb_id,
                    "name": matched_ref.name,
                    "similarity": cluster.stashdb_match_score,
                    "image_url": getattr(matched_ref, "image_url", None),
                },
            })
        else:
            # Try to find local performers by exact name match
            name_fallbacks = _find_local_performers_by_name(matched_ref.name)
            found_name_fallback = False
            for fallback in name_fallbacks:
                if _performer_has_conflicting_stashdb_id(
                    fallback["id"],
                    matched_ref.stashdb_id,
                    matched_ref.source_endpoint,
                ):
                    continue

                found_name_fallback = True
                stashdb_performer_ids.add(fallback["id"])
                linked_cluster = get_cluster_for_performer(fallback["id"])
                already_linked = linked_cluster is not None and linked_cluster.id != cluster_id

                stashdb_suggestions.append({
                    "performer_id": fallback["id"],
                    "performer_name": fallback["name"],
                    "performer_image": f"/performer/{fallback['id']}/image",
                    "stashdb_image_url": getattr(matched_ref, "image_url", None),
                    "scene_count": 0,
                    "image_count": 0,
                    "co_occurrence_count": 0,
                    "confidence": "stashdb_name",
                    "solo_scene_count": 0,
                    "solo_image_count": 0,
                    "already_linked": already_linked,
                    "stashdb_match": {
                        "stashdb_id": matched_ref.stashdb_id,
                        "name": matched_ref.name,
                        "similarity": cluster.stashdb_match_score,
                        "image_url": getattr(matched_ref, "image_url", None),
                    },
                })

            # No local performer exists at all — inject a "create from
            # StashDB" suggestion so the user can still see who this is
            # and create the performer directly from the link dialog.
            if not found_name_fallback:
                stashdb_suggestions.append({
                    "performer_id": None,
                    "performer_name": matched_ref.name,
                    "performer_image": getattr(matched_ref, "image_url", None),
                    "stashdb_image_url": getattr(matched_ref, "image_url", None),
                    "scene_count": 0,
                    "image_count": 0,
                    "co_occurrence_count": 0,
                    "confidence": "stashdb_only",
                    "solo_scene_count": 0,
                    "solo_image_count": 0,
                    "already_linked": False,
                    "stashdb_match": {
                        "stashdb_id": matched_ref.stashdb_id,
                        "name": matched_ref.name,
                        "disambiguation": matched_ref.disambiguation,
                        "similarity": cluster.stashdb_match_score,
                        "image_url": getattr(matched_ref, "image_url", None),
                    },
                })

    if total_entities == 0 and not stashdb_suggestions:
        return {
            "suggestions": [],
            "cluster_scene_count": 0,
            "cluster_image_count": 0,
            "cluster_entity_count": 0,
        }

    # Step 3: Cross-DB query — read Stash SQLite for performer co-occurrence
    performer_counts = _query_stash_co_occurrence(scene_ids, image_ids) if total_entities > 0 else {}

    # Step 4: Rank by co-occurrence count, compute confidence
    co_occurrence_suggestions = []
    for pid, data in sorted(
        performer_counts.items(), key=lambda x: -x[1]["co_occurrence_count"]
    ):
        if pid in stashdb_performer_ids:
            # Already included as a StashDB suggestion — merge counts
            # and upgrade confidence to indicate dual match
            for s in stashdb_suggestions:
                if s["performer_id"] == pid:
                    s["scene_count"] = data["scene_count"]
                    s["image_count"] = data["image_count"]
                    s["co_occurrence_count"] = data["co_occurrence_count"]
                    s["solo_scene_count"] = data["solo_scene_count"]
                    s["solo_image_count"] = data["solo_image_count"]
                    # Mark as dual match: StashDB embedding + scene co-occurrence
                    if data["co_occurrence_count"] > 0 and s["confidence"] in ("stashdb", "stashdb_name"):
                        s["confidence"] = "stashdb_scene"
            continue

        ratio = data["co_occurrence_count"] / total_entities if total_entities > 0 else 0
        confidence = "high" if ratio > 0.7 else "possible" if ratio > 0.3 else "suggested"

        # Check if already linked to another cluster (query our PG DB)
        linked_cluster = get_cluster_for_performer(pid)
        already_linked = linked_cluster is not None and linked_cluster.id != cluster_id

        # Build performer image URL.  The raw image column from the Stash
        # SQLite DB is a filesystem path (e.g. "generated/performers/42/image.jpg")
        # which browsers cannot load directly.  Use the Stash HTTP endpoint
        # ``/performer/{id}/image`` which serves the image correctly.
        raw_image = data["image"]
        if raw_image and ("/" in raw_image or "\\" in raw_image):
            # Raw filesystem path — replace with Stash API URL
            performer_img = f"/performer/{pid}/image"
        else:
            performer_img = raw_image or f"/performer/{pid}/image"

        co_occurrence_suggestions.append({
            "performer_id": pid,
            "performer_name": data["name"],
            "performer_image": performer_img,
            "scene_count": data["scene_count"],
            "image_count": data["image_count"],
            "co_occurrence_count": data["co_occurrence_count"],
            "confidence": confidence,
            "solo_scene_count": data["solo_scene_count"],
            "solo_image_count": data["solo_image_count"],
            "already_linked": already_linked,
        })

    # StashDB suggestions come first (highest priority)
    suggestions = stashdb_suggestions + co_occurrence_suggestions

    return {
        "suggestions": suggestions,
        "cluster_scene_count": len(scene_ids),
        "cluster_image_count": len(image_ids),
        "cluster_entity_count": total_entities,
    }


# ---------------------------------------------------------------------------
# Entity face data
# ---------------------------------------------------------------------------

@router.get("/entity/{entity_type}/{entity_id}")
async def get_entity_faces(
    entity_type: str,
    entity_id: int,
    db: Session = Depends(get_db),
):
    """Get face detection tracks for an entity (scene or image)."""
    _require_plugin_active(db, PLUGIN_NAME)
    if entity_type not in ("scene", "image"):
        raise HTTPException(status_code=400, detail="entity_type must be 'scene' or 'image'")

    tracks = get_entity_tracks(entity_type, entity_id)
    result = []
    for t in tracks:
        # Look up cluster info for face tracks
        cluster_id = None
        cluster_status = None
        performer_id = None
        if t.label == "face" and hasattr(t, "embeddings") and t.embeddings:
            first_emb = t.embeddings[0] if t.embeddings else None
            if first_emb and first_emb.cluster_id:
                cluster_id = first_emb.cluster_id
                cluster_obj = get_cluster_by_id(cluster_id)
                if cluster_obj:
                    cluster_status = cluster_obj.status
                    performer_id = cluster_obj.performer_id

        result.append({
            "id": t.id,
            "label": t.label,
            "bbox": list(t.bbox) if t.bbox else [],
            "score": t.score,
            "start_s": t.start_s,
            "end_s": t.end_s,
            "cluster_id": cluster_id,
            "cluster_status": cluster_status,
            "performer_id": performer_id,
        })

    return {"tracks": result}


# ═══════════════════════════════════════════════════════════════════════════
# StashDB Performer Reference Embeddings
# ═══════════════════════════════════════════════════════════════════════════

from stash_ai_server.db.stashdb_store import (
    _sanitize_text as _stashdb_sanitize_text,
    _sanitize_aliases as _stashdb_sanitize_aliases,
    bulk_upsert_stashdb_refs,
    clear_local_performer_id_async as _clear_local_performer_id_async,
    delete_pack as _delete_pack,
    delete_pack_async as _delete_pack_async,
    backfill_stashdb_matches_async as _backfill_stashdb_matches_async,
    delete_ref_async as _delete_ref_async,
    find_nearest_stashdb_ref,
    find_nearest_stashdb_ref_async,
    get_ref_by_id as _get_ref_by_id,
    get_stats_async as _get_stats_async,
    list_packs_async as _list_packs_async,
    list_refs_async as _list_refs_async,
    set_local_performer_id_async as _set_local_performer_id_async,
    upsert_stashdb_ref,
)
from stash_ai_server.models.detections import StashDBPerformerRef


@router.post("/stashdb/import")
async def import_saie_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload and import a ``.saie`` embedding pack.

    Returns a streamed NDJSON response with progress updates so the
    frontend can display a live progress bar.  Each line is a JSON
    object with at least ``{"phase", "progress", "total"}``.  The final
    line contains the full result summary.
    """
    import json
    import time
    import zipfile
    import tempfile

    _require_plugin_active(db, PLUGIN_NAME)

    content = await file.read()
    if len(content) < 100:
        raise HTTPException(status_code=400, detail="File is too small to be a valid .saie archive")

    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip archive")

    # Validate structure
    names = set(zf.namelist())
    if "manifest.json" not in names:
        raise HTTPException(status_code=400, detail="Missing manifest.json in archive")
    if "performers.jsonl" not in names:
        raise HTTPException(status_code=400, detail="Missing performers.jsonl in archive")
    if "centroids.npy" not in names:
        raise HTTPException(status_code=400, detail="Missing centroids.npy in archive")

    # Parse manifest
    try:
        manifest = json.loads(zf.read("manifest.json"))
    except (json.JSONDecodeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid manifest.json: {exc}")

    version = manifest.get("version", 1)
    if version > 1:
        raise HTTPException(status_code=400, detail=f"Unsupported format version: {version}")

    embedder = manifest.get("embedder", "unknown")
    embedding_dim = manifest.get("embedding_dim", 512)
    source_endpoint = manifest.get("source_endpoint")
    pack_id = manifest.get("pack_id") or manifest.get("source", "unknown")

    # Parse centroids
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp.write(zf.read("centroids.npy"))
        tmp_path = tmp.name

    try:
        centroids = np.load(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid centroids.npy: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if centroids.ndim != 2 or centroids.shape[1] != embedding_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected centroids shape (N, {embedding_dim}), got {centroids.shape}",
        )

    # Parse performers
    performers_raw = zf.read("performers.jsonl").decode("utf-8").strip().split("\n")
    performers = []
    for line in performers_raw:
        if line.strip():
            try:
                performers.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(performers) != centroids.shape[0]:
        raise HTTPException(
            status_code=400,
            detail=f"Performer count ({len(performers)}) != centroid rows ({centroids.shape[0]})",
        )

    # ── Stream progress as NDJSON ─────────────────────────────────────
    async def _stream_import():
        t_start = time.monotonic()

        # Phase 1: Resolve local performers
        yield json.dumps({"phase": "resolve", "message": "Resolving local performers...", "progress": 0, "total": len(performers)}) + "\n"
        t0 = time.monotonic()
        all_stashdb_ids = [p["stashdb_id"] for p in performers if "stashdb_id" in p]
        local_map = await asyncio.to_thread(
            _batch_find_local_performers_by_stashdb_ids,
            all_stashdb_ids, source_endpoint,
        )
        t_resolve = time.monotonic() - t0
        _log.info("SAIE import: local performer resolve took %.2fs for %d ids (%d matched)",
                  t_resolve, len(all_stashdb_ids), len(local_map))
        yield json.dumps({"phase": "resolve", "message": f"Resolved {len(local_map)} local performers", "progress": len(performers), "total": len(performers), "elapsed": round(t_resolve, 2)}) + "\n"

        # Phase 2: Build rows (pre-sanitize text, bulk-convert centroids)
        yield json.dumps({"phase": "prepare", "message": "Preparing data...", "progress": 0, "total": len(performers)}) + "\n"
        t0 = time.monotonic()
        # Convert the full centroid matrix to a Python list-of-lists in one
        # numpy call — far faster than calling .tolist() per row (avoids
        # creating 51.2M individual Python float objects in a Python loop).
        centroid_lists = centroids.astype(np.float32).tolist()
        sanitized_ep = _stashdb_sanitize_text(source_endpoint)
        sanitized_pack = _stashdb_sanitize_text(pack_id)

        upsert_rows: list[dict] = []
        for i, perf in enumerate(performers):
            sid = perf.get("stashdb_id")
            if not sid:
                continue
            local = local_map.get(sid)
            upsert_rows.append({
                "stashdb_id": sid,
                "name": _stashdb_sanitize_text(perf["name"]) or sid,
                "centroid": centroid_lists[i],
                "embedder": embedder,
                "disambiguation": _stashdb_sanitize_text(perf.get("disambiguation")),
                "aliases": _stashdb_sanitize_aliases(perf.get("aliases")),
                "sample_count": perf.get("sample_count", 0),
                "quality_score": perf.get("quality_score"),
                "source_endpoint": sanitized_ep,
                "pack_id": sanitized_pack,
                "local_performer_id": local["id"] if local else None,
                "image_url": perf.get("image_url"),
            })
        t_prepare = time.monotonic() - t0
        _log.info("SAIE import: row preparation took %.2fs for %d rows", t_prepare, len(upsert_rows))
        yield json.dumps({"phase": "prepare", "message": f"Prepared {len(upsert_rows)} rows", "progress": len(performers), "total": len(performers), "elapsed": round(t_prepare, 2)}) + "\n"

        # Phase 3: Bulk upsert with progress
        upsert_total = len(upsert_rows)
        yield json.dumps({"phase": "upsert", "message": "Inserting into database...", "progress": 0, "total": upsert_total}) + "\n"
        t0 = time.monotonic()

        # Use a queue to bridge the sync callback into async streaming
        progress_queue: asyncio.Queue[tuple[int, int]] = asyncio.Queue()

        def _on_progress(done: int, total: int):
            progress_queue.put_nowait((done, total))

        # Run bulk upsert in a thread, streaming progress as it arrives
        upsert_task = asyncio.get_running_loop().run_in_executor(
            None, lambda: bulk_upsert_stashdb_refs(upsert_rows, on_progress=_on_progress),
        )

        last_yield = 0
        while not upsert_task.done():
            try:
                done, total = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                # Only yield if meaningful change (avoid flooding)
                if done - last_yield >= 500 or done >= total:
                    yield json.dumps({"phase": "upsert", "message": f"Inserted {done}/{total}", "progress": done, "total": total}) + "\n"
                    last_yield = done
            except asyncio.TimeoutError:
                pass

        imported, updated, errors = await upsert_task

        # Drain any remaining progress messages
        while not progress_queue.empty():
            try:
                progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        t_upsert = time.monotonic() - t0
        _log.info("SAIE import: DB upsert took %.2fs — imported=%d updated=%d errors=%d",
                  t_upsert, imported, updated, errors)
        yield json.dumps({"phase": "upsert", "message": f"Upsert complete: {imported} new, {updated} updated", "progress": upsert_total, "total": upsert_total, "elapsed": round(t_upsert, 2)}) + "\n"

        # Phase 4: Backfill
        yield json.dumps({"phase": "backfill", "message": "Matching face clusters...", "progress": 0, "total": 0}) + "\n"
        t0 = time.monotonic()
        clusters_matched = 0
        try:
            clusters_matched = await _backfill_stashdb_matches_async()
        except Exception:
            _log.exception("StashDB backfill failed after import")
        t_backfill = time.monotonic() - t0
        _log.info("SAIE import: backfill took %.2fs — %d clusters matched", t_backfill, clusters_matched)

        t_total = time.monotonic() - t_start
        _log.info("SAIE import: total %.2fs (resolve=%.2fs prepare=%.2fs upsert=%.2fs backfill=%.2fs)",
                  t_total, t_resolve, t_prepare, t_upsert, t_backfill)

        # Final result line
        yield json.dumps({
            "phase": "done",
            "status": "ok",
            "imported": imported,
            "updated": updated,
            "errors": errors,
            "total_in_pack": len(performers),
            "pack_id": pack_id,
            "embedder": embedder,
            "clusters_matched": clusters_matched,
            "timing": {
                "resolve_s": round(t_resolve, 2),
                "prepare_s": round(t_prepare, 2),
                "upsert_s": round(t_upsert, 2),
                "backfill_s": round(t_backfill, 2),
                "total_s": round(t_total, 2),
            },
        }) + "\n"

    return StreamingResponse(_stream_import(), media_type="application/x-ndjson")


@router.get("/stashdb/packs")
async def list_stashdb_packs(db: Session = Depends(get_db)):
    """List imported StashDB embedding packs."""
    _require_plugin_active(db, PLUGIN_NAME)
    packs = await _list_packs_async()
    return {"packs": packs}


@router.delete("/stashdb/packs/{pack_id}")
async def delete_stashdb_pack(pack_id: str, db: Session = Depends(get_db)):
    """Remove all performer refs from a pack."""
    _require_plugin_active(db, PLUGIN_NAME)
    count = await _delete_pack_async(pack_id)
    return {"status": "ok", "deleted": count}


@router.get("/stashdb/refs")
async def list_stashdb_refs(
    search: str | None = Query(None),
    pack_id: str | None = Query(None),
    has_local_performer: bool | None = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    sort: str = Query("name"),
    sort_dir: str = Query("asc"),
    db: Session = Depends(get_db),
):
    """List StashDB performer references with pagination and search."""
    _require_plugin_active(db, PLUGIN_NAME)
    refs, total = await _list_refs_async(
        search=search, pack_id=pack_id,
        has_local_performer=has_local_performer,
        page=page, per_page=per_page,
        sort=sort, sort_dir=sort_dir,
    )
    serialized_refs = []
    for r in refs:
        verified_local = await _get_verified_local_performer_for_ref(r, repair=True)
        serialized_refs.append({
            "id": r.id,
            "stashdb_id": r.stashdb_id,
            "name": r.name,
            "disambiguation": r.disambiguation,
            "aliases": r.aliases,
            "sample_count": r.sample_count,
            "quality_score": r.quality_score,
            "embedder": r.embedder,
            "source_endpoint": r.source_endpoint,
            "pack_id": r.pack_id,
            "local_performer_id": verified_local["id"] if verified_local else None,
            "image_url": getattr(r, "image_url", None),
            "extra_endpoints": r.extra_endpoints,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        })
    return {
        "refs": serialized_refs,
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.get("/stashdb/refs/{ref_id}")
async def get_stashdb_ref(ref_id: int, db: Session = Depends(get_db)):
    """Get a single StashDB performer reference."""
    _require_plugin_active(db, PLUGIN_NAME)
    ref = _get_ref_by_id(ref_id)
    if ref is None:
        raise HTTPException(status_code=404, detail="StashDB ref not found")
    verified_local = await _get_verified_local_performer_for_ref(ref, repair=True)
    return {
        "id": ref.id,
        "stashdb_id": ref.stashdb_id,
        "name": ref.name,
        "disambiguation": ref.disambiguation,
        "aliases": ref.aliases,
        "sample_count": ref.sample_count,
        "quality_score": ref.quality_score,
        "embedder": ref.embedder,
        "source_endpoint": ref.source_endpoint,
        "pack_id": ref.pack_id,
        "local_performer_id": verified_local["id"] if verified_local else None,
        "image_url": getattr(ref, "image_url", None),
        "extra_endpoints": ref.extra_endpoints,
        "created_at": ref.created_at.isoformat() if ref.created_at else None,
        "updated_at": ref.updated_at.isoformat() if ref.updated_at else None,
    }


@router.delete("/stashdb/refs/{ref_id}")
async def delete_stashdb_ref(ref_id: int, db: Session = Depends(get_db)):
    """Delete a single StashDB performer reference."""
    _require_plugin_active(db, PLUGIN_NAME)
    ok = await _delete_ref_async(ref_id)
    if not ok:
        raise HTTPException(status_code=404, detail="StashDB ref not found")
    return {"status": "ok"}


class CreatePerformerFromRefRequest(BaseModel):
    """Request to create a local Stash performer from a StashDB reference."""
    cluster_id: int | None = None  # optionally link a face cluster
    use_stashdb_data: bool = True  # whether to attach stash_ids to the new performer


@router.post("/stashdb/refs/{ref_id}/create-performer")
async def create_performer_from_ref(
    ref_id: int,
    payload: CreatePerformerFromRefRequest,
    db: Session = Depends(get_db),
):
    """Create a local Stash performer from a StashDB reference.

    If the reference already has a ``local_performer_id``, that existing
    performer is used instead of creating a duplicate.  When
    ``cluster_id`` is supplied, the cluster is linked to the performer and
    the performer is retroactively applied to all related scenes/images.
    """
    _require_plugin_active(db, PLUGIN_NAME)
    ref = _get_ref_by_id(ref_id)
    if ref is None:
        raise HTTPException(status_code=404, detail="StashDB ref not found")

    verified_local = await _get_verified_local_performer_for_ref(ref, repair=True)
    performer_id = verified_local["id"] if verified_local else None

    if performer_id is None:
        if payload.use_stashdb_data and ref.source_endpoint:
            performer_id = stash_api.create_performer(
                ref.name,
                stash_ids=[{"stash_id": ref.stashdb_id, "endpoint": ref.source_endpoint}],
                disambiguation=ref.disambiguation,
            )
        else:
            performer_id = stash_api.create_performer(ref.name)

        if performer_id is None:
            # Create failed (likely "already exists") — find existing performer by name
            existing = _find_local_performers_by_name(ref.name)
            if existing:
                performer_id = existing[0]["id"]
                _log.info(
                    "create_performer_from_ref: using existing performer '%s' (id=%s) for ref %d",
                    existing[0]["name"], performer_id, ref_id,
                )
                # Attach stash_ids if available
                if payload.use_stashdb_data and ref.source_endpoint and ref.stashdb_id:
                    try:
                        stash_api.update_performer(
                            performer_id,
                            stash_ids=[{"stash_id": ref.stashdb_id, "endpoint": ref.source_endpoint}],
                        )
                    except Exception:
                        _log.debug("Failed to set stash_ids on existing performer %s", performer_id)
            else:
                raise HTTPException(status_code=500, detail="Failed to create performer in Stash")

        await _set_local_performer_id_async(ref_id, performer_id)

    if payload.use_stashdb_data:
        try:
            await _hydrate_performer_from_stashdb_ref(performer_id, ref)
        except Exception:
            _log.exception(
                "Failed to hydrate performer %s from StashDB ref %s",
                performer_id, ref_id,
            )

    # Link cluster if specified
    applied: dict = {}
    if payload.cluster_id:
        link_performer(payload.cluster_id, performer_id, label=ref.name)
        applied = _apply_performer_to_cluster_entities(payload.cluster_id, performer_id)

    return {
        "status": "ok",
        "performer_id": performer_id,
        "performer_name": ref.name,
        "stashdb_id": ref.stashdb_id,
        **applied,
    }


def _find_local_performer_by_stashdb_id(stashdb_id: str, endpoint: str | None) -> dict | None:
    """Look up a local performer by exact stash-id match in Stash SQLite."""
    performers_table = stash_db.get_stash_table("performers", required=False)
    stash_ids_table = stash_db.get_stash_table("performer_stash_ids", required=False)
    if performers_table is None or stash_ids_table is None:
        return None
    session_factory = stash_db.get_stash_sessionmaker()
    if session_factory is None:
        return None
    try:
        normalized_endpoint = _normalize_stash_endpoint(endpoint)
        with session_factory() as session:
            rows = session.execute(
                sa.select(
                    performers_table.c.id,
                    performers_table.c.name,
                    stash_ids_table.c.endpoint,
                )
                .select_from(
                    performers_table.join(
                        stash_ids_table,
                        performers_table.c.id == stash_ids_table.c.performer_id,
                    )
                )
                .where(stash_ids_table.c.stash_id == stashdb_id)
            ).fetchall()

        if normalized_endpoint is not None:
            rows = [row for row in rows if _normalize_stash_endpoint(row[2]) == normalized_endpoint]

        performer_rows = {(int(row[0]), row[1]) for row in rows}
        if len(performer_rows) != 1:
            return None

        performer_id, performer_name = next(iter(performer_rows))
        return {"id": performer_id, "name": performer_name}
    except Exception:
        _log.debug("Could not look up performer by stash id '%s'", stashdb_id, exc_info=True)
        return None


def _batch_find_local_performers_by_stashdb_ids(
    stashdb_ids: list[str],
    endpoint: str | None,
) -> dict[str, dict]:
    """Batch lookup: stashdb_id -> {id, name} for all matching local performers.

    Returns only stashdb_ids that map to exactly one local performer (same
    logic as the single-lookup version).
    """
    if not stashdb_ids:
        return {}
    performers_table = stash_db.get_stash_table("performers", required=False)
    stash_ids_table = stash_db.get_stash_table("performer_stash_ids", required=False)
    if performers_table is None or stash_ids_table is None:
        return {}
    perf_id_col = performers_table.c.get("id")
    perf_name_col = performers_table.c.get("name")
    if perf_id_col is None or perf_name_col is None:
        return {}
    session_factory = stash_db.get_stash_sessionmaker()
    if session_factory is None:
        return {}

    normalized_endpoint = _normalize_stash_endpoint(endpoint)
    result: dict[str, dict] = {}

    try:
        with session_factory() as session:
            # SQLite has a variable limit; chunk if needed
            chunk_size = 900
            for i in range(0, len(stashdb_ids), chunk_size):
                chunk = stashdb_ids[i : i + chunk_size]
                rows = session.execute(
                    sa.select(
                        stash_ids_table.c.stash_id,
                        perf_id_col,
                        perf_name_col,
                        stash_ids_table.c.endpoint,
                    )
                    .select_from(
                        stash_ids_table.join(
                            performers_table,
                            perf_id_col == stash_ids_table.c.performer_id,
                        )
                    )
                    .where(stash_ids_table.c.stash_id.in_(chunk))
                ).fetchall()

                # Group by stashdb_id
                by_stashdb_id: dict[str, list[tuple]] = {}
                for row in rows:
                    sid = row[0]
                    by_stashdb_id.setdefault(sid, []).append(row)

                for sid, sid_rows in by_stashdb_id.items():
                    if normalized_endpoint is not None:
                        sid_rows = [
                            r for r in sid_rows
                            if _normalize_stash_endpoint(r[3]) == normalized_endpoint
                        ]
                    performer_set = {(int(r[1]), r[2]) for r in sid_rows}
                    if len(performer_set) == 1:
                        pid, pname = next(iter(performer_set))
                        result[sid] = {"id": pid, "name": pname}
    except Exception:
        _log.debug("Batch stashdb_id lookup failed", exc_info=True)

    return result


def _find_local_performers_by_name(name: str) -> list[dict]:
    """Look up all local performers by case-insensitive name."""
    performers_table = stash_db.get_stash_table("performers", required=False)
    if performers_table is None:
        return []
    session_factory = stash_db.get_stash_sessionmaker()
    if session_factory is None:
        return []
    try:
        with session_factory() as session:
            perf_id_col = performers_table.c.get("id")
            perf_name_col = performers_table.c.get("name")
            if perf_id_col is None or perf_name_col is None:
                return []
            rows = session.execute(
                sa.select(perf_id_col, perf_name_col)
                .where(sa.func.lower(perf_name_col) == name.lower().strip())
                .order_by(perf_id_col.asc())
            ).fetchall()
            return [{"id": int(row[0]), "name": row[1]} for row in rows]
    except Exception:
        _log.debug("Could not look up performer by name '%s'", name)
        return []


def _normalize_stash_endpoint(endpoint: str | None) -> str | None:
    if not endpoint:
        return None
    return endpoint.strip().rstrip("/").lower()


def _performer_has_conflicting_stashdb_id(
    performer_id: int,
    stashdb_id: str,
    endpoint: str | None,
) -> bool:
    """Return True if the performer already has a different StashDB stash-id."""
    stash_ids_table = stash_db.get_stash_table("performer_stash_ids", required=False)
    if stash_ids_table is None:
        return False
    session_factory = stash_db.get_stash_sessionmaker()
    if session_factory is None:
        return False
    try:
        normalized_endpoint = _normalize_stash_endpoint(endpoint)
        with session_factory() as session:
            rows = session.execute(
                sa.select(stash_ids_table.c.endpoint, stash_ids_table.c.stash_id)
                .where(stash_ids_table.c.performer_id == performer_id)
            ).fetchall()
        for row in rows:
            row_endpoint = _normalize_stash_endpoint(row[0])
            if normalized_endpoint is not None and row_endpoint != normalized_endpoint:
                continue
            if row[1] and str(row[1]) != stashdb_id:
                return True
        return False
    except Exception:
        _log.debug("Could not inspect stash ids for performer %s", performer_id, exc_info=True)
        return False


async def _get_verified_local_performer_for_ref(
    ref: StashDBPerformerRef,
    *,
    repair: bool,
) -> dict | None:
    """Return the verified local performer for a StashDB ref, if any."""
    exact_match = _find_local_performer_by_stashdb_id(ref.stashdb_id, ref.source_endpoint)
    current_local_id = ref.local_performer_id

    if exact_match is not None:
        if current_local_id != exact_match["id"] and repair:
            await _set_local_performer_id_async(ref.id, exact_match["id"])
            ref.local_performer_id = exact_match["id"]
        return exact_match

    if current_local_id is not None and repair:
        await _clear_local_performer_id_async(ref.id)
        ref.local_performer_id = None

    return None


@router.post("/stashdb/dismiss/{cluster_id}")
async def dismiss_stashdb_match(cluster_id: int, db: Session = Depends(get_db)):
    """Clear the StashDB match suggestion for a cluster."""
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    with get_session_local()() as sess:
        sess.execute(
            sa.update(FaceCluster)
            .where(FaceCluster.id == cluster_id)
            .values(stashdb_match_id=None, stashdb_match_score=None)
        )
        sess.commit()

    return {"status": "ok", "cluster_id": cluster_id}


@router.get("/stashdb/stats")
async def get_stashdb_stats(db: Session = Depends(get_db)):
    """Return aggregate statistics about imported StashDB refs."""
    _require_plugin_active(db, PLUGIN_NAME)
    stats = await _get_stats_async()
    return stats


@router.get("/stashdb/match/{cluster_id}")
async def get_stashdb_matches_for_cluster(
    cluster_id: int,
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Find the best StashDB ref matches for a cluster's centroid."""
    _require_plugin_active(db, PLUGIN_NAME)
    cluster = get_cluster_by_id(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    if cluster.centroid is None:
        return {"matches": []}

    centroid = list(cluster.centroid) if not isinstance(cluster.centroid, list) else cluster.centroid
    matches = await find_nearest_stashdb_ref_async(centroid, limit=limit, min_similarity=0.50)
    return {
        "matches": [
            {"ref_id": m[0], "stashdb_id": m[1], "name": m[2], "similarity": round(m[3], 4)}
            for m in matches
        ]
    }


# ---------------------------------------------------------------------------
# Presence-based auto-linking
# ---------------------------------------------------------------------------

class AutoLinkByPresenceRequest(BaseModel):
    min_scenes: int = 3
    """Cluster must appear in at least this many distinct scenes."""
    min_images: int = 10
    """Or at least this many distinct images (OR logic with min_scenes)."""
    min_co_occurrence_ratio: float = 0.40
    """The top-suggested performer must appear in at least this fraction of the cluster's entities."""
    require_stashdb_match: bool = True
    """Only auto-link clusters that already have a StashDB embedding match (any score)."""
    dry_run: bool = False
    """If True, compute candidates but do not commit any links."""


@router.post("/clusters/auto-link-by-presence")
async def auto_link_by_presence(
    payload: AutoLinkByPresenceRequest,
    db: Session = Depends(get_db),
):
    """Auto-link unidentified clusters to performers using presence + co-occurrence logic.

    A cluster qualifies when it has been seen enough (scene/image count thresholds)
    AND the top co-occurring performer appears in a high enough fraction of those
    entities.  Optionally restricted to clusters that already have a StashDB
    embedding match (which serves as a cross-validation signal).

    This is useful for performers who appear frequently — repeated co-occurrence
    at scale is strong evidence even without a high-confidence StashDB score.
    """
    _require_plugin_active(db, PLUGIN_NAME)

    # Fetch all unidentified cluster IDs
    all_ids = list_cluster_ids(status="unidentified")
    if not all_ids:
        return {"status": "ok", "linked": 0, "skipped": 0, "candidates": [], "dry_run": payload.dry_run}

    # Pre-fetch entity counts in bulk to filter by presence threshold quickly
    with get_session_local()() as sess:
        from stash_ai_server.models.detections import DetectionTrack
        count_rows = sess.execute(
            sa.select(
                DetectionTrack.cluster_id,
                DetectionTrack.entity_type,
                sa.func.count(sa.distinct(DetectionTrack.entity_id)).label("cnt"),
            )
            .where(
                DetectionTrack.cluster_id.in_(all_ids),
                DetectionTrack.cluster_id.isnot(None),
            )
            .group_by(DetectionTrack.cluster_id, DetectionTrack.entity_type)
        ).all()

    entity_counts: dict[int, dict[str, int]] = {}
    for row in count_rows:
        entity_counts.setdefault(row.cluster_id, {"scene": 0, "image": 0})
        entity_counts[row.cluster_id][row.entity_type] = row.cnt

    # Filter: presence threshold (OR logic)
    qualifying_ids = [
        cid for cid in all_ids
        if entity_counts.get(cid, {}).get("scene", 0) >= payload.min_scenes
        or entity_counts.get(cid, {}).get("image", 0) >= payload.min_images
    ]
    if not qualifying_ids:
        return {"status": "ok", "linked": 0, "skipped": len(all_ids), "candidates": [], "dry_run": payload.dry_run}

    # Optionally filter to clusters that have a StashDB match
    if payload.require_stashdb_match:
        with get_session_local()() as sess:
            rows = sess.execute(
                sa.select(FaceCluster.id)
                .where(
                    FaceCluster.id.in_(qualifying_ids),
                    FaceCluster.stashdb_match_id.isnot(None),
                    FaceCluster.stashdb_suggestion_rejected.isnot(sa.true()),
                )
            ).scalars().all()
        qualifying_ids = list(rows)

    if not qualifying_ids:
        return {"status": "ok", "linked": 0, "skipped": len(all_ids), "candidates": [], "dry_run": payload.dry_run}

    # Get top suggestions (co-occurrence + stashdb) for qualifying clusters
    suggestions = _compute_batch_top_suggestions(qualifying_ids)

    linked = 0
    skipped = 0
    candidates: list[dict] = []

    for cid in qualifying_ids:
        suggestion = suggestions.get(cid)
        ec = entity_counts.get(cid, {})
        scene_cnt = ec.get("scene", 0)
        image_cnt = ec.get("image", 0)
        total_entities = scene_cnt + image_cnt

        if suggestion is None:
            skipped += 1
            continue

        ratio = suggestion.get("co_occurrence_ratio", 0.0)
        performer_id = suggestion.get("performer_id")
        performer_name = suggestion.get("performer_name", "")
        confidence = suggestion.get("confidence", "")

        # Reject if co-occurrence is too low, unless it's a stashdb-only match
        # with no co-occurrence data (meaning the cluster has no entity overlap yet)
        if ratio < payload.min_co_occurrence_ratio and confidence != "stashdb":
            skipped += 1
            continue

        # For stashdb-only suggestions with no local performer, we would need
        # to create a performer — skip in auto mode to avoid noise
        if performer_id is None:
            skipped += 1
            continue

        candidates.append({
            "cluster_id": cid,
            "performer_id": performer_id,
            "performer_name": performer_name,
            "confidence": confidence,
            "co_occurrence_ratio": ratio,
            "scene_count": scene_cnt,
            "image_count": image_cnt,
            "total_entities": total_entities,
        })

        if not payload.dry_run:
            try:
                cluster = get_cluster_by_id(cid)
                if cluster is None or cluster.status != "unidentified":
                    skipped += 1
                    continue
                link_performer(cid, performer_id, label=performer_name or None)
                _apply_performer_to_cluster_entities(cid, performer_id)
                linked += 1
                _log.debug(
                    "Auto-link-by-presence: cluster %d → performer %d (%s) "
                    "scenes=%d images=%d ratio=%.2f",
                    cid, performer_id, performer_name, scene_cnt, image_cnt, ratio,
                )
            except Exception:
                _log.exception("Auto-link-by-presence: failed to link cluster %d", cid)
                skipped += 1

    return {
        "status": "ok",
        "linked": linked if not payload.dry_run else 0,
        "skipped": skipped,
        "candidates": candidates,
        "dry_run": payload.dry_run,
    }


# ---------------------------------------------------------------------------
# Cluster merge suggestions (same performer)
# ---------------------------------------------------------------------------

class MergeByPerformerRequest(BaseModel):
    min_centroid_similarity: float = 0.60
    """Two clusters for the same performer only merge if their centroids are at least this similar.
    Lower values allow merging visually distinct appearances; higher values keep only near-identical clusters."""
    dry_run: bool = False
    """If True, return merge candidates but do not commit merges."""


@router.post("/clusters/merge-by-performer")
async def merge_clusters_by_performer(
    payload: MergeByPerformerRequest,
    db: Session = Depends(get_db),
):
    """Find and optionally merge clusters that share the same identified performer.

    When a performer has been linked to multiple clusters (e.g. due to different
    lighting / angle creating spurious splits), this endpoint merges the smaller
    clusters into the largest one — provided their centroids are visually similar
    enough.  Clusters whose centroids differ significantly (e.g. genuinely
    different eras of appearance) are left separate.

    Design note: Having multiple clusters per performer is sometimes intentional
    (different hairstyles, significant appearance changes over time).  Use
    ``min_centroid_similarity`` to control how aggressively clusters merge:
    - ~0.90 → only near-identical clusters merge (safe default for cleanup)
    - ~0.70 → moderately different appearances are merged
    - ~0.50 → aggressive merging; may combine genuinely distinct looks
    """
    _require_plugin_active(db, PLUGIN_NAME)

    # Find all identified clusters grouped by performer_id
    with get_session_local()() as sess:
        rows = sess.execute(
            sa.select(FaceCluster.id, FaceCluster.performer_id, FaceCluster.sample_count, FaceCluster.centroid)
            .where(
                FaceCluster.status == "identified",
                FaceCluster.performer_id.isnot(None),
                FaceCluster.centroid.isnot(None),
            )
        ).all()

    # Group by performer
    by_performer: dict[int, list[dict]] = {}
    for row in rows:
        by_performer.setdefault(row.performer_id, []).append({
            "id": row.id,
            "sample_count": row.sample_count or 0,
            "centroid": np.array(row.centroid, dtype=np.float32),
        })

    merge_groups: list[dict] = []
    merged = 0
    skipped_low_sim = 0

    for performer_id, clusters in by_performer.items():
        if len(clusters) < 2:
            continue

        # Sort descending by sample_count — largest is the surviving cluster
        clusters_sorted = sorted(clusters, key=lambda c: c["sample_count"], reverse=True)
        surviving = clusters_sorted[0]
        s_vec = surviving["centroid"]
        s_norm = np.linalg.norm(s_vec)
        if s_norm > 0:
            s_vec = s_vec / s_norm

        to_merge: list[dict] = []
        to_skip: list[dict] = []

        for other in clusters_sorted[1:]:
            o_vec = other["centroid"]
            o_norm = np.linalg.norm(o_vec)
            if o_norm > 0:
                o_vec = o_vec / o_norm
            sim = float(np.dot(s_vec, o_vec))

            if sim >= payload.min_centroid_similarity:
                to_merge.append({**other, "centroid_similarity": round(sim, 4)})
            else:
                to_skip.append({**other, "centroid_similarity": round(sim, 4)})
                skipped_low_sim += 1

        if not to_merge:
            continue

        merge_groups.append({
            "performer_id": performer_id,
            "surviving_cluster_id": surviving["id"],
            "surviving_sample_count": surviving["sample_count"],
            "absorbing": [
                {"cluster_id": c["id"], "sample_count": c["sample_count"], "centroid_similarity": c["centroid_similarity"]}
                for c in to_merge
            ],
            "skipped_low_similarity": [
                {"cluster_id": c["id"], "sample_count": c["sample_count"], "centroid_similarity": c["centroid_similarity"]}
                for c in to_skip
            ],
        })

        if not payload.dry_run:
            for c in to_merge:
                try:
                    merge_clusters(surviving["id"], c["id"])
                    merged += 1
                    _log.info(
                        "Merged cluster %d into %d (performer %d, sim=%.3f)",
                        c["id"], surviving["id"], performer_id, c["centroid_similarity"],
                    )
                except Exception:
                    _log.exception(
                        "Failed to merge cluster %d into %d", c["id"], surviving["id"]
                    )

    return {
        "status": "ok",
        "merge_groups": merge_groups,
        "merged": merged if not payload.dry_run else 0,
        "skipped_low_similarity": skipped_low_sim,
        "dry_run": payload.dry_run,
    }


# ---------------------------------------------------------------------------
# Cluster diagnostics / settings tuning
# ---------------------------------------------------------------------------

@router.get("/diagnostics/cluster-stats")
async def get_cluster_diagnostics(
    db: Session = Depends(get_db),
    max_clusters: int = Query(200, ge=1, le=1000, description="Max clusters to include in inter-cluster analysis"),
    include_embedding_stats: bool = Query(True, description="Compute intra-cluster embedding similarity stats (slower)"),
):
    """Return diagnostic statistics to help tune face recognition settings.

    Covers:
    - **Intra-cluster similarity**: how consistent embeddings are within each cluster.
      High values (>0.90) mean tight, confident clusters.
      Low values (<0.70) may indicate a cluster contains multiple people.
    - **Inter-cluster nearest-neighbor similarity**: how close clusters are to each other.
      Values near or above ``auto_threshold`` indicate clusters that might merge during
      the next scan.
    - **Threshold simulation**: at a given ``auto_threshold``, how many unidentified
      clusters would absorb into an existing cluster instead of creating a new one.
    - **StashDB match score distribution**: where your existing matches fall.
    - **Co-occurrence confidence distribution**: how often the top suggested performer
      dominates each cluster's entity list.
    """
    _require_plugin_active(db, PLUGIN_NAME)

    # --- 1. Fetch all active clusters (excluding merged_away) ---
    with get_session_local()() as sess:
        cluster_rows = sess.execute(
            sa.select(
                FaceCluster.id,
                FaceCluster.status,
                FaceCluster.sample_count,
                FaceCluster.centroid,
                FaceCluster.stashdb_match_score,
                FaceCluster.stashdb_suggestion_rejected,
                FaceCluster.performer_id,
            )
            .where(FaceCluster.status != "merged_away")
            .order_by(FaceCluster.id)
            .limit(max_clusters)
        ).all()

    cluster_ids = [r.id for r in cluster_rows]
    centroid_by_id = {
        r.id: np.array(r.centroid, dtype=np.float32)
        for r in cluster_rows
        if r.centroid is not None
    }

    # Normalise centroids
    normed: dict[int, np.ndarray] = {}
    for cid, vec in centroid_by_id.items():
        n = np.linalg.norm(vec)
        normed[cid] = vec / n if n > 0 else vec

    # --- 2. Summary by status ---
    status_counts: Counter[str] = Counter(r.status for r in cluster_rows)

    # --- 3. Intra-cluster embedding similarity (optional, heavier) ---
    intra_stats: list[dict] = []
    if include_embedding_stats and cluster_ids:
        ec_all = await asyncio.to_thread(_compute_intra_cluster_stats, cluster_ids)
        intra_stats = ec_all

    # --- 4. Inter-cluster nearest-neighbour (pairwise centroid similarity) ---
    ids_with_centroid = list(normed.keys())
    nn_pairs: list[dict] = []
    threshold_sim_counts: dict[str, int] = {}

    if len(ids_with_centroid) >= 2:
        vecs = np.array([normed[cid] for cid in ids_with_centroid], dtype=np.float32)
        sim_matrix = vecs @ vecs.T  # shape (N, N)
        n = len(ids_with_centroid)

        # Nearest neighbour for each cluster (excluding self)
        for i in range(n):
            row = sim_matrix[i].copy()
            row[i] = -1.0  # exclude self
            j = int(np.argmax(row))
            nn_pairs.append({
                "cluster_a": ids_with_centroid[i],
                "cluster_b": ids_with_centroid[j],
                "similarity": round(float(sim_matrix[i, j]), 4),
            })

        # Threshold simulation: for each threshold T, how many cluster pairs
        # would overlap (sim >= T)?  These would be candidates to merge during scan.
        for thresh_10 in range(3, 10):
            t = thresh_10 / 10.0
            key = f"{t:.1f}"
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i, j] >= t:
                        count += 1
            threshold_sim_counts[key] = count

    # --- 5. StashDB match score distribution ---
    stashdb_scores = [
        r.stashdb_match_score for r in cluster_rows
        if r.stashdb_match_score is not None
    ]
    stashdb_buckets = _make_histogram(stashdb_scores, [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)])

    # --- 6. Entity counts per cluster ---
    entity_count_data = {}
    if cluster_ids:
        with get_session_local()() as sess:
            from stash_ai_server.models.detections import DetectionTrack
            ec_rows = sess.execute(
                sa.select(
                    DetectionTrack.cluster_id,
                    DetectionTrack.entity_type,
                    sa.func.count(sa.distinct(DetectionTrack.entity_id)).label("cnt"),
                )
                .where(
                    DetectionTrack.cluster_id.in_(cluster_ids),
                    DetectionTrack.cluster_id.isnot(None),
                )
                .group_by(DetectionTrack.cluster_id, DetectionTrack.entity_type)
            ).all()
        for ec in ec_rows:
            entity_count_data.setdefault(ec.cluster_id, {"scene": 0, "image": 0})
            entity_count_data[ec.cluster_id][ec.entity_type] = ec.cnt

    # --- 7. Embedding-level quality stats per cluster ---
    emb_stats_by_cluster: dict[int, dict] = {}
    if cluster_ids:
        from stash_ai_server.models.detections import FaceEmbedding as FE
        with get_session_local()() as sess:
            emb_rows = sess.execute(
                sa.select(
                    FE.cluster_id,
                    sa.func.count(FE.id).label("emb_count"),
                    sa.func.avg(FE.norm).label("mean_norm"),
                    sa.func.min(FE.norm).label("min_norm"),
                    sa.func.max(FE.norm).label("max_norm"),
                    sa.func.avg(FE.score).label("mean_score"),
                    sa.func.min(FE.score).label("min_score"),
                    sa.func.max(FE.score).label("max_score"),
                )
                .where(FE.cluster_id.in_(cluster_ids))
                .group_by(FE.cluster_id)
            ).all()
        for er in emb_rows:
            emb_stats_by_cluster[er.cluster_id] = {
                "emb_count": er.emb_count,
                "mean_norm": round(float(er.mean_norm), 2),
                "min_norm": round(float(er.min_norm), 2),
                "max_norm": round(float(er.max_norm), 2),
                "mean_score": round(float(er.mean_score), 4),
                "min_score": round(float(er.min_score), 4),
                "max_score": round(float(er.max_score), 4),
            }

    # --- 8. Build per-cluster detail ---
    cluster_details = []
    for r in cluster_rows:
        detail: dict = {
            "cluster_id": r.id,
            "status": r.status,
            "sample_count": r.sample_count or 0,
            "has_centroid": r.id in centroid_by_id,
            "scene_count": entity_count_data.get(r.id, {}).get("scene", 0),
            "image_count": entity_count_data.get(r.id, {}).get("image", 0),
            "stashdb_match_score": round(r.stashdb_match_score, 4) if r.stashdb_match_score else None,
            "stashdb_suggestion_rejected": bool(r.stashdb_suggestion_rejected),
            "performer_id": r.performer_id,
        }
        cluster_details.append(detail)

    # Attach intra-cluster stats and embedding stats
    intra_by_id = {s["cluster_id"]: s for s in intra_stats}
    for d in cluster_details:
        s = intra_by_id.get(d["cluster_id"])
        if s:
            d["intra_cluster"] = {k: v for k, v in s.items() if k != "cluster_id"}
        es = emb_stats_by_cluster.get(d["cluster_id"])
        if es:
            d["embedding_stats"] = es

    return {
        "summary": {
            "total_clusters": len(cluster_rows),
            "by_status": dict(status_counts),
        },
        "reference_profile": _compute_reference_profile(cluster_details, emb_stats_by_cluster),
        "inter_cluster": {
            "nearest_neighbors": sorted(nn_pairs, key=lambda x: x["similarity"], reverse=True)[:50],
            "threshold_simulation": [
                {
                    "auto_threshold": float(t),
                    "cluster_pairs_would_overlap": count,
                    "note": "pairs whose centroids are >= threshold (candidates to merge during scan)",
                }
                for t, count in sorted(threshold_sim_counts.items())
            ],
        },
        "stashdb_match_scores": {
            "count": len(stashdb_scores),
            "mean": round(float(np.mean(stashdb_scores)), 4) if stashdb_scores else None,
            "histogram": stashdb_buckets,
        },
        "stashdb_threshold_analysis": _stashdb_threshold_analysis(cluster_details),
        "clusters": cluster_details,
    }


def _compute_reference_profile(
    cluster_details: list[dict],
    emb_stats_by_cluster: dict[int, dict],
) -> dict:
    """Derive recommended thresholds from three ground-truth populations.

    Populations (after manual curation):
    - **identified** – linked to a performer → positive ground truth
    - **ignored** – user marked as bad/junk → negative quality ground truth
    - **stashdb_rejected** – user said StashDB suggestion was wrong
      → negative match ground truth (good quality, wrong identity)
    - **unidentified** – not yet curated → unknown, used for context only

    Thresholds are set at the statistical gap between these populations
    so they adapt as more clusters are curated.
    """
    identified = [
        c for c in cluster_details
        if c.get("status") == "identified" and c.get("performer_id") is not None
    ]
    ignored = [
        c for c in cluster_details
        if c.get("status") == "ignored"
    ]
    # Clusters where the user explicitly rejected the StashDB suggestion
    # (still unidentified but the stashdb match was wrong)
    stashdb_rejected = [
        c for c in cluster_details
        if c.get("stashdb_suggestion_rejected")
        and c.get("stashdb_match_score") is not None
        and c.get("status") != "identified"
    ]
    unidentified = [
        c for c in cluster_details
        if c.get("status") == "unidentified"
    ]

    def _pop_stats(population: list[dict], label: str) -> dict:
        """Compute aggregate statistics for a cluster population."""
        intra_sims = [
            c["intra_cluster"]["mean_sim"]
            for c in population
            if c.get("intra_cluster", {}).get("mean_sim") is not None
        ]
        norms = [
            emb_stats_by_cluster[c["cluster_id"]]["mean_norm"]
            for c in population
            if c["cluster_id"] in emb_stats_by_cluster
        ]
        min_norms = [
            emb_stats_by_cluster[c["cluster_id"]]["min_norm"]
            for c in population
            if c["cluster_id"] in emb_stats_by_cluster
        ]
        scores = [
            emb_stats_by_cluster[c["cluster_id"]]["mean_score"]
            for c in population
            if c["cluster_id"] in emb_stats_by_cluster
        ]
        min_scores = [
            emb_stats_by_cluster[c["cluster_id"]]["min_score"]
            for c in population
            if c["cluster_id"] in emb_stats_by_cluster
        ]
        stashdb = [
            c["stashdb_match_score"]
            for c in population
            if c.get("stashdb_match_score") is not None
        ]
        sample_counts = [c.get("sample_count", 0) for c in population]

        def _pct(vals: list[float], p: int) -> float | None:
            if not vals:
                return None
            return round(float(np.percentile(vals, p)), 4)

        return {
            "label": label,
            "count": len(population),
            "intra_similarity": {
                "count": len(intra_sims),
                "mean": round(float(np.mean(intra_sims)), 4) if intra_sims else None,
                "p5": _pct(intra_sims, 5),
                "p25": _pct(intra_sims, 25),
                "median": _pct(intra_sims, 50),
                "p75": _pct(intra_sims, 75),
            },
            "embedding_norm": {
                "count": len(norms),
                "mean_of_means": round(float(np.mean(norms)), 2) if norms else None,
                "p5_of_mins": _pct(min_norms, 5),
                "p25_of_mins": _pct(min_norms, 25),
                "median_of_means": _pct(norms, 50),
            },
            "detection_score": {
                "count": len(scores),
                "mean_of_means": round(float(np.mean(scores)), 4) if scores else None,
                "p5_of_mins": _pct(min_scores, 5),
                "p25_of_mins": _pct(min_scores, 25),
                "median_of_means": _pct(scores, 50),
            },
            "stashdb_score": {
                "count": len(stashdb),
                "mean": round(float(np.mean(stashdb)), 4) if stashdb else None,
                "min": round(min(stashdb), 4) if stashdb else None,
                "max": round(max(stashdb), 4) if stashdb else None,
                "p5": _pct(stashdb, 5),
                "p25": _pct(stashdb, 25),
                "p75": _pct(stashdb, 75),
                "p95": _pct(stashdb, 95),
            },
            "sample_count": {
                "mean": round(float(np.mean(sample_counts)), 1) if sample_counts else None,
                "median": _pct(sample_counts, 50),
                "single_sample_pct": round(
                    100 * sum(1 for s in sample_counts if s <= 1) / max(len(sample_counts), 1), 1
                ),
            },
        }

    id_stats = _pop_stats(identified, "identified")
    ign_stats = _pop_stats(ignored, "ignored")
    rej_stats = _pop_stats(stashdb_rejected, "stashdb_rejected")
    un_stats = _pop_stats(unidentified, "unidentified")

    # --- Derive recommended thresholds ---
    recommendations: dict[str, dict] = {}

    # ---- Quality thresholds (norm & detection score) ----
    # Strategy: use the gap between identified (good) and ignored (bad).
    # If we have ignored clusters, place the threshold at the midpoint of
    # the identified floor (p5) and the ignored ceiling (p95), so it
    # cleanly separates the two populations.
    # If no ignored clusters yet, fall back to identified-only (p5).
    id_min_norms_p5 = id_stats["embedding_norm"]["p5_of_mins"]
    ign_norms_p95 = None
    if ign_stats["embedding_norm"]["count"]:
        # p95 of ignored clusters' mean norms = the upper end of "bad"
        ign_mean_norms = [
            emb_stats_by_cluster[c["cluster_id"]]["mean_norm"]
            for c in ignored
            if c["cluster_id"] in emb_stats_by_cluster
        ]
        if ign_mean_norms:
            ign_norms_p95 = round(float(np.percentile(ign_mean_norms, 95)), 2)

    if id_min_norms_p5 is not None:
        if ign_norms_p95 is not None and ign_norms_p95 < float(id_min_norms_p5):
            # Gap exists: place threshold at midpoint
            mid = (float(id_min_norms_p5) + ign_norms_p95) / 2
            suggested = round(mid * 2) / 2  # round to nearest 0.5
            basis = (
                f"midpoint of identified p5 min norm ({id_min_norms_p5}) "
                f"and ignored p95 mean norm ({ign_norms_p95})"
            )
        else:
            suggested = round(float(id_min_norms_p5) * 2) / 2
            basis = f"p5 of identified clusters' min embedding norms = {id_min_norms_p5}"
            if ign_norms_p95 is not None:
                basis += f" (ignored p95 = {ign_norms_p95}, no clear gap — using identified floor)"
        recommendations["face_min_embedding_norm"] = {
            "suggested": suggested,
            "basis": basis,
        }

    id_min_scores_p5 = id_stats["detection_score"]["p5_of_mins"]
    ign_scores_p95 = None
    if ign_stats["detection_score"]["count"]:
        ign_mean_scores = [
            emb_stats_by_cluster[c["cluster_id"]]["mean_score"]
            for c in ignored
            if c["cluster_id"] in emb_stats_by_cluster
        ]
        if ign_mean_scores:
            ign_scores_p95 = round(float(np.percentile(ign_mean_scores, 95)), 4)

    if id_min_scores_p5 is not None:
        if ign_scores_p95 is not None and ign_scores_p95 < float(id_min_scores_p5):
            mid = (float(id_min_scores_p5) + ign_scores_p95) / 2
            suggested = round(mid, 2)
            basis = (
                f"midpoint of identified p5 min score ({id_min_scores_p5}) "
                f"and ignored p95 mean score ({ign_scores_p95})"
            )
        else:
            suggested = round(float(id_min_scores_p5), 2)
            basis = f"p5 of identified clusters' min detection scores = {id_min_scores_p5}"
            if ign_scores_p95 is not None:
                basis += f" (ignored p95 = {ign_scores_p95}, no clear gap)"
        recommendations["face_min_detection_score"] = {
            "suggested": suggested,
            "basis": basis,
        }

    # ---- StashDB auto-link threshold ----
    # Strategy: use the gap between confirmed-correct stashdb matches
    # (identified clusters with stashdb scores) and rejected matches
    # (stashdb_suggestion_rejected clusters).
    id_stashdb_min = id_stats["stashdb_score"]["min"]
    rej_stashdb_max = rej_stats["stashdb_score"]["max"]

    if id_stashdb_min is not None:
        if rej_stashdb_max is not None and float(rej_stashdb_max) < float(id_stashdb_min):
            # Clear gap between worst correct match and best wrong match
            mid = (float(id_stashdb_min) + float(rej_stashdb_max)) / 2
            suggested = round(mid, 2)
            basis = (
                f"midpoint of lowest confirmed match ({id_stashdb_min}) "
                f"and highest rejected match ({rej_stashdb_max})"
            )
        elif rej_stashdb_max is not None:
            # Overlap: some rejected scores are >= confirmed scores.
            # Use the confirmed min as floor (be conservative).
            suggested = round(float(id_stashdb_min), 2)
            basis = (
                f"lowest confirmed match ({id_stashdb_min}); WARNING: "
                f"rejected matches go up to {rej_stashdb_max} (overlap!). "
                f"Manual review needed for scores in [{rej_stashdb_max}, {id_stashdb_min}]"
            )
        else:
            # No rejected data yet — just use identified min with safety margin
            suggested = round(max(0.50, float(id_stashdb_min) - 0.02), 2)
            basis = (
                f"lowest confirmed match ({id_stashdb_min}) minus 0.02 safety margin. "
                f"Reject some StashDB suggestions to refine this further."
            )
        recommendations["stashdb_auto_link_threshold"] = {
            "suggested": suggested,
            "basis": basis,
        }

    # ---- Cluster matching thresholds ----
    id_intra_p5 = id_stats["intra_similarity"]["p5"]
    id_intra_median = id_stats["intra_similarity"]["median"]

    if id_intra_p5 is not None:
        suggested = round(float(id_intra_p5) * 0.75, 2)
        suggested = max(0.25, min(0.50, suggested))
        recommendations["face_match_review_threshold"] = {
            "suggested": suggested,
            "basis": f"p5 of identified intra_similarity = {id_intra_p5}, scaled to 75% and clamped [0.25, 0.50]",
        }

    review_sugg = recommendations.get("face_match_review_threshold", {}).get("suggested")
    if id_intra_median is not None and review_sugg is not None:
        suggested = round((float(review_sugg) + float(id_intra_median)) / 2, 2)
        suggested = max(0.35, min(0.60, suggested))
        recommendations["face_match_auto_threshold"] = {
            "suggested": suggested,
            "basis": (
                f"midpoint of review_threshold ({review_sugg}) and "
                f"identified intra_similarity median ({id_intra_median}), clamped [0.35, 0.60]"
            ),
        }

    # ---- Curation progress ----
    total = len(cluster_details)
    curated = len(identified) + len(ignored) + len(stashdb_rejected)
    uncurated_unidentified = len([
        c for c in unidentified
        if not c.get("stashdb_suggestion_rejected")
    ])

    return {
        "populations": {
            "identified": id_stats,
            "ignored": ign_stats,
            "stashdb_rejected": rej_stats,
            "unidentified": un_stats,
        },
        "recommended_thresholds": recommendations,
        "curation_progress": {
            "total_clusters": total,
            "identified": len(identified),
            "ignored": len(ignored),
            "stashdb_rejected": len(stashdb_rejected),
            "uncurated": uncurated_unidentified,
            "curated_pct": round(100 * curated / max(total, 1), 1),
        },
        "workflow_hint": (
            "Link correct matches (identified), mark bad crops as ignored, "
            "reject wrong StashDB suggestions. Each action improves threshold "
            "recommendations. Aim for >=30 identified + >=10 ignored for "
            "reliable quality thresholds, and >=5 rejected for StashDB tuning."
        ),
    }


def _stashdb_threshold_analysis(cluster_details: list[dict]) -> list[dict]:
    """Show what would be auto-linked at various StashDB thresholds.

    For each candidate threshold, reports how many unlinked clusters with a
    StashDB match score would qualify for auto-linking, broken down by
    cluster quality (intra-cluster mean_sim).
    """
    unlinked_with_score = [
        c for c in cluster_details
        if c.get("stashdb_match_score") is not None and c.get("performer_id") is None
    ]
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    result = []
    for t in thresholds:
        qualifying = [c for c in unlinked_with_score if c["stashdb_match_score"] >= t]
        # Break down by quality
        good = sum(
            1 for c in qualifying
            if c.get("intra_cluster", {}).get("mean_sim") is not None
            and c["intra_cluster"]["mean_sim"] >= 0.55
        )
        questionable = sum(
            1 for c in qualifying
            if c.get("intra_cluster", {}).get("mean_sim") is not None
            and c["intra_cluster"]["mean_sim"] < 0.55
        )
        no_stats = sum(
            1 for c in qualifying
            if c.get("intra_cluster", {}).get("mean_sim") is None
        )
        result.append({
            "threshold": t,
            "would_auto_link": len(qualifying),
            "good_quality": good,
            "questionable_quality": questionable,
            "no_quality_stats": no_stats,
            "cluster_ids": [c["cluster_id"] for c in qualifying],
        })
    return result


def _compute_intra_cluster_stats(cluster_ids: list[int]) -> list[dict]:
    """For each cluster, compute mean/min/max cosine similarity between its exemplar embeddings."""
    results = []
    for cid in cluster_ids:
        exemplars = get_cluster_exemplars(cid)
        if len(exemplars) < 2:
            results.append({
                "cluster_id": cid,
                "embedding_count": len(exemplars),
                "mean_sim": None,
                "min_sim": None,
                "max_sim": None,
            })
            continue

        vecs = []
        for e in exemplars:
            if e.embedding is None:
                continue
            v = np.array(e.embedding, dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            vecs.append(v)

        if len(vecs) < 2:
            results.append({
                "cluster_id": cid,
                "embedding_count": len(exemplars),
                "mean_sim": None,
                "min_sim": None,
                "max_sim": None,
            })
            continue

        mat = np.array(vecs) @ np.array(vecs).T
        # Upper triangle only (exclude diagonal)
        triu = mat[np.triu_indices(len(vecs), k=1)]
        results.append({
            "cluster_id": cid,
            "embedding_count": len(exemplars),
            "mean_sim": round(float(np.mean(triu)), 4),
            "min_sim": round(float(np.min(triu)), 4),
            "max_sim": round(float(np.max(triu)), 4),
        })
    return results


def _make_histogram(values: list[float], buckets: list[tuple[float, float]]) -> list[dict]:
    """Bin float values into labelled histogram buckets."""
    result = []
    for lo, hi in buckets:
        label = f"{lo:.1f}–{hi:.2g}" if hi < 1.01 else f"{lo:.1f}+"
        count = sum(1 for v in values if lo <= v < hi)
        result.append({"range": label, "count": count})
    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_routes() -> APIRouter:
    """Return the face cluster API router."""
    return router
