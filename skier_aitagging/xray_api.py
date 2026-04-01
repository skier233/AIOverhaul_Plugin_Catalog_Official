"""REST API endpoints for the X-Ray debug overlay.

Mounted under ``/xray/`` via the plugin router system — final URL is
``/api/v1/plugins/skier_aitagging/xray/...``.

Two endpoints:
  GET  /scenes/{scene_id}/data          – scene overlay payload
  POST /scenes/{scene_id}/capture-frame – snapshot visible faces/tags
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import sqlalchemy as sa
from sqlalchemy import select

from stash_ai_server.core import config as app_config
from stash_ai_server.db.session import get_session_local
from stash_ai_server.db.ai_results_store import get_scene_timespans
from stash_ai_server.models.detections import (
    DetectionTrack,
    FaceCluster,
    FaceEmbedding,
)
from stash_ai_server.models.ai_results import AIModelRun

from . import stash_handler

_log = logging.getLogger(__name__)

router = APIRouter(prefix="/skier_aitagging/xray", tags=["skier_aitagging_xray"])

SERVICE_NAME = "AI_Tagging"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_scene_detection_tracks(scene_id: int) -> list[DetectionTrack]:
    """Return all detection tracks for a given scene (entity_type='scene')."""
    with get_session_local()() as session:
        stmt = (
            select(DetectionTrack)
            .join(AIModelRun, DetectionTrack.run_id == AIModelRun.id)
            .where(
                AIModelRun.service == SERVICE_NAME,
                AIModelRun.entity_type == "scene",
                AIModelRun.entity_id == scene_id,
                DetectionTrack.label == "face",
            )
            .order_by(DetectionTrack.start_s.asc().nullslast())
        )
        return list(session.execute(stmt).scalars().all())


def _get_scene_face_embeddings(scene_id: int) -> list[FaceEmbedding]:
    """Return all face embeddings for a scene."""
    with get_session_local()() as session:
        stmt = (
            select(FaceEmbedding)
            .where(
                FaceEmbedding.entity_type == "scene",
                FaceEmbedding.entity_id == scene_id,
            )
            .order_by(FaceEmbedding.timestamp_s.asc().nullslast())
        )
        return list(session.execute(stmt).scalars().all())


def _resolve_performer_names(cluster_ids: set[int]) -> dict[int, dict[str, Any]]:
    """Batch-resolve cluster → {performer_id, performer_label, status} for a set of cluster IDs.

    When a cluster has ``performer_id`` set but no ``label``, looks up the
    performer name from the Stash SQLite database so the overlay can show
    real names instead of numeric IDs.
    """
    if not cluster_ids:
        return {}

    with get_session_local()() as session:
        stmt = (
            select(FaceCluster)
            .where(FaceCluster.id.in_(cluster_ids))
        )
        clusters = list(session.execute(stmt).scalars().all())

    # Collect performer IDs that need name resolution
    needs_name: dict[int, list[int]] = {}  # performer_id -> [cluster_ids]
    result: dict[int, dict[str, Any]] = {}
    for c in clusters:
        result[c.id] = {
            "performer_id": c.performer_id,
            "performer_label": c.label,
            "cluster_status": c.status,
        }
        if c.performer_id and not c.label:
            needs_name.setdefault(c.performer_id, []).append(c.id)

    # Batch-resolve missing names from Stash DB
    if needs_name:
        try:
            from stash_ai_server.utils import stash_db
            performers_table = stash_db.get_stash_table("performers", required=False)
            if performers_table is not None:
                name_col = performers_table.c.get("name")
                id_col = performers_table.c.get("id")
                if name_col is not None and id_col is not None:
                    session_factory = stash_db.get_stash_sessionmaker()
                    if session_factory is not None:
                        with session_factory() as sess:
                            rows = sess.execute(
                                sa.select(id_col, name_col)
                                .where(id_col.in_(list(needs_name.keys())))
                            ).all()
                        for pid, pname in rows:
                            for cid in needs_name.get(int(pid), []):
                                result[cid]["performer_label"] = pname
        except Exception:
            _log.debug("Could not resolve performer names from Stash DB", exc_info=True)

    return result


def _build_tag_id_to_name_map() -> dict[str, str]:
    """Invert stash_handler's AI tag cache to map tag_id (str) → tag_name."""
    cache = stash_handler.get_ai_tags_cache()
    # cache is {name: id, ...} → invert to {str(id): name}
    return {str(v): k for k, v in cache.items()}


def _format_tag_timespans(
    raw_buckets: dict | None,
) -> dict[str, list[dict[str, Any]]]:
    """Transform DB timespan buckets into the frontend ``tag_timespans`` shape.

    DB format:  {category: {tag_id_str: [{start, end, confidence}, ...]}}
    Frontend:   {category: [{tag_id, tag_name, category, spans: [{start, end, confidence}]}]}
    """
    if not raw_buckets:
        return {}

    tag_name_map = _build_tag_id_to_name_map()
    result: dict[str, list[dict[str, Any]]] = {}

    for category, tag_dict in raw_buckets.items():
        cat_key = category or "uncategorized"
        entries: list[dict[str, Any]] = []
        for tag_id_str, span_list in (tag_dict or {}).items():
            tag_name = tag_name_map.get(tag_id_str, f"tag_{tag_id_str}")
            try:
                tag_id_int = int(tag_id_str)
            except (ValueError, TypeError):
                tag_id_int = 0
            entries.append({
                "tag_id": tag_id_int,
                "tag_name": tag_name,
                "category": cat_key,
                "spans": [
                    {
                        "start": s.get("start", 0),
                        "end": s.get("end", 0),
                        "confidence": s.get("confidence", 1.0),
                    }
                    for s in span_list
                ],
            })
        if entries:
            result[cat_key] = entries

    return result


# ---------------------------------------------------------------------------
# GET /scenes/{scene_id}/data
# ---------------------------------------------------------------------------

@router.get("/scenes/{scene_id}/data")
async def get_scene_xray_data(scene_id: int) -> dict[str, Any]:
    """Return the full X-Ray overlay payload for a scene.

    Response shape::

        {
            "scene_id": 123,
            "face_detections": [...],
            "tracks": [...],
            "tag_timespans": { category: [TagSpanEntry, ...] }
        }
    """
    tracks_raw, embeddings_raw, timespan_buckets = await asyncio.gather(
        asyncio.to_thread(_get_scene_detection_tracks, scene_id),
        asyncio.to_thread(_get_scene_face_embeddings, scene_id),
        asyncio.to_thread(get_scene_timespans, service=SERVICE_NAME, scene_id=scene_id),
    )

    # Collect unique cluster_ids from tracks and embeddings for batch performer resolution
    cluster_ids: set[int] = set()
    for t in tracks_raw:
        if t.cluster_id is not None:
            cluster_ids.add(t.cluster_id)
    for e in embeddings_raw:
        if e.cluster_id is not None:
            cluster_ids.add(e.cluster_id)

    cluster_info = await asyncio.to_thread(_resolve_performer_names, cluster_ids)

    # Build tracks array
    tracks: list[dict[str, Any]] = []
    for t in tracks_raw:
        ci = cluster_info.get(t.cluster_id, {}) if t.cluster_id else {}
        tracks.append({
            "track_id": t.id,
            "bbox": list(t.bbox) if t.bbox else [],
            "score": t.score,
            "start_s": t.start_s,
            "end_s": t.end_s,
            "keyframes": t.keyframes,
            "cluster_id": t.cluster_id,
            "performer_id": ci.get("performer_id"),
            "performer_label": ci.get("performer_label"),
            "cluster_status": ci.get("cluster_status"),
        })

    # Build face_detections array (one entry per embedding)
    face_detections: list[dict[str, Any]] = []
    for e in embeddings_raw:
        ci = cluster_info.get(e.cluster_id, {}) if e.cluster_id else {}
        face_detections.append({
            "embedding_id": e.id,
            "track_id": e.track_id,
            "bbox": list(e.bbox) if e.bbox else [],
            "timestamp_s": e.timestamp_s,
            "score": e.score,
            "is_exemplar": e.is_exemplar,
            "cluster_id": e.cluster_id,
            "performer_id": ci.get("performer_id"),
            "performer_label": ci.get("performer_label"),
            "cluster_status": ci.get("cluster_status"),
        })

    # Build tag_timespans
    tag_timespans = _format_tag_timespans(timespan_buckets)

    return {
        "scene_id": scene_id,
        "face_detections": face_detections,
        "tracks": tracks,
        "tag_timespans": tag_timespans,
    }


# ---------------------------------------------------------------------------
# POST /scenes/{scene_id}/capture-frame
# ---------------------------------------------------------------------------

class _CapturedFace(BaseModel):
    bbox: list[float]
    score: float
    performer_label: str | None = None
    cluster_id: int | None = None


class _CapturedTag(BaseModel):
    tag_name: str
    category: str
    confidence: float = 1.0


class CaptureFrameBody(BaseModel):
    timestamp_s: float
    tags: list[_CapturedTag] = []
    faces: list[_CapturedFace] = []


@router.post("/scenes/{scene_id}/capture-frame")
async def capture_frame(scene_id: int, body: CaptureFrameBody) -> dict[str, Any]:
    """Snapshot the currently visible faces and tags at a timestamp.

    Saves:
    - ``<data_dir>/xray_captures/<scene_id>/<timestamp>.json`` — metadata
    - ``<data_dir>/xray_captures/<scene_id>/<timestamp>.png``  — video frame image
    """
    capture_dir = Path(app_config.data_dir) / "xray_captures" / str(scene_id)

    # Extract the video frame at the requested timestamp
    frame_bytes: bytes | None = None
    try:
        from . import face_api as _face_api
        from stash_ai_server.utils.stash_api import stash_api as _stash_api
        video_path = await asyncio.to_thread(_face_api._get_scene_file_path, scene_id)
        if video_path and Path(video_path).is_file():
            frame_bytes = await asyncio.to_thread(
                _face_api._extract_video_frame, video_path, body.timestamp_s,
            )
    except Exception:
        _log.debug("Could not extract video frame for xray capture scene %s", scene_id, exc_info=True)

    def _write() -> tuple[int, int]:
        capture_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "scene_id": scene_id,
            "timestamp_s": body.timestamp_s,
            "tags": [t.model_dump() for t in body.tags],
            "faces": [f.model_dump() for f in body.faces],
            "has_image": frame_bytes is not None,
        }
        base = f"{body.timestamp_s:.3f}"
        json_path = capture_dir / f"{base}.json"
        data = json.dumps(payload, indent=2)
        json_path.write_text(data, encoding="utf-8")

        img_size = 0
        if frame_bytes:
            img_path = capture_dir / f"{base}.png"
            img_path.write_bytes(frame_bytes)
            img_size = len(frame_bytes)

        return len(data), img_size

    try:
        json_size, img_size = await asyncio.to_thread(_write)
    except Exception as exc:
        _log.exception("Failed to write xray capture for scene %s", scene_id)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "json_size_bytes": json_size,
        "image_size_bytes": img_size,
        "has_image": img_size > 0,
    }


# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

def register_routes() -> APIRouter:
    """Return the router to be mounted by the plugin loader."""
    return router
