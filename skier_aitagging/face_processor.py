"""Face detection / embedding processing pipeline.

This module sits between the response parser (which produces typed detections
and region results) and the detection store (which persists to PostgreSQL).

Three main entry points:
- ``process_image_detections``  — single-image path
- ``process_video_detections``  — scene / video path
- ``build_tracks``              — IoU-based frame-to-frame track builder
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

import numpy as np

from .models import (
    Detection,
    EmbeddingResult,
    FrameEmbedding,
    ParsedFrameData,
    ParsedImageData,
    RegionResult,
    TrackCandidate,
)
from .response_parser import parse_embeddings

from stash_ai_server.db.detection_store import (
    cleanup_stale_detections_async,
    count_cluster_embeddings,
    create_cluster,
    find_nearest_cluster,
    get_cluster_by_id,
    get_cluster_exemplars,
    merge_clusters,
    store_detection_track,
    store_face_embedding,
    try_add_exemplar,
    update_cluster_centroid,
)
from stash_ai_server.db.session import get_session_local
from stash_ai_server.utils.stash_api import stash_api as _stash_api

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FACE_LABEL = "face"
_FACE_DETECTION_CATEGORIES = ("face_detections",)
_FACE_EMBEDDING_CATEGORY = "face_embeddings"
_REGION_PREFIX = "regions__"

DEFAULT_IOU_THRESHOLD = 0.50
DEFAULT_KEYFRAME_IOU_THRESHOLD = 0.80
DEFAULT_MAX_TRACK_GAP_FRAMES = 3
DEFAULT_DEDUP_THRESHOLD = 0.85
DEFAULT_MAX_EMBEDDINGS_PER_TRACK = 10
DEFAULT_AUTO_THRESHOLD = 0.50
DEFAULT_REVIEW_THRESHOLD = 0.40
DEFAULT_MAX_EXEMPLARS = 10
# Quality floor: minimum thresholds for an embedding to be used as an
# exemplar or to seed a new cluster.  Below-quality faces can still be
# matched to *existing* clusters (growing appearance counts) but won't
# become exemplars that degrade centroids or create spurious new clusters.
DEFAULT_MIN_EMBEDDING_NORM = 18.0
DEFAULT_MIN_DETECTION_SCORE = 0.65
# Hard floor: absolute minimum to accept an embedding at all.  Below this
# the detection is almost certainly a false positive or completely unusable.
# Anything above the hard floor but below the quality floor gets stored and
# matched but never becomes an exemplar or seeds a new cluster.
DEFAULT_HARD_MIN_EMBEDDING_NORM = 10.0
DEFAULT_HARD_MIN_DETECTION_SCORE = 0.30
# Maximum embeddings to persist per cluster per scene.  Limits DB bloat
# when many short tracks map to the same identity.  The best embeddings
# (by score * norm) are kept.
DEFAULT_MAX_EMBEDDINGS_PER_CLUSTER = 20


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection over Union for two [x1, y1, x2, y2] bboxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _l2_normalise(vec: list[float] | np.ndarray) -> tuple[np.ndarray, float]:
    """Return (normalised_vector, original_norm)."""
    v = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    return v, norm


# ---------------------------------------------------------------------------
# Track builder  (Phase 1 of video pipeline)
# ---------------------------------------------------------------------------

class _OpenTrack:
    """Mutable accumulator used during track building."""

    __slots__ = (
        "label", "detector", "best_bbox", "best_score",
        "last_bbox", "start_s", "end_s", "last_frame_idx",
        "keyframes", "embeddings",
    )

    def __init__(
        self,
        label: str,
        detector: str,
        bbox: list[float],
        score: float,
        timestamp_s: float,
        frame_idx: float,
    ) -> None:
        self.label = label
        self.detector = detector
        self.best_bbox = list(bbox)
        self.best_score = score
        self.last_bbox = list(bbox)
        self.start_s = timestamp_s
        self.end_s = timestamp_s
        self.last_frame_idx = frame_idx
        self.keyframes: list[dict[str, Any]] = [{"t": timestamp_s, "bbox": list(bbox)}]
        self.embeddings: list[FrameEmbedding] = []

    def update(
        self,
        bbox: list[float],
        score: float,
        timestamp_s: float,
        frame_idx: float,
        keyframe_iou_threshold: float,
    ) -> None:
        self.end_s = timestamp_s
        self.last_frame_idx = frame_idx
        if score > self.best_score:
            self.best_score = score
            self.best_bbox = list(bbox)
        # Only store a new keyframe if bbox shifted significantly
        if _iou(self.last_bbox, bbox) < keyframe_iou_threshold:
            self.keyframes.append({"t": timestamp_s, "bbox": list(bbox)})
        self.last_bbox = list(bbox)

    def to_candidate(self) -> TrackCandidate:
        return TrackCandidate(
            label=self.label,
            best_bbox=self.best_bbox,
            best_score=self.best_score,
            detector=self.detector,
            start_s=self.start_s,
            end_s=self.end_s,
            keyframes=self.keyframes if len(self.keyframes) > 1 else None,
            embeddings=list(self.embeddings),
        )


def build_tracks(
    frames: Sequence[ParsedFrameData],
    frame_interval: float,
    *,
    label: str = FACE_LABEL,
    detection_categories: Sequence[str] = _FACE_DETECTION_CATEGORIES,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    keyframe_iou_threshold: float = DEFAULT_KEYFRAME_IOU_THRESHOLD,
    max_gap_frames: int = DEFAULT_MAX_TRACK_GAP_FRAMES,
) -> list[TrackCandidate]:
    """IoU-based frame-to-frame track builder.

    Returns a list of completed :class:`TrackCandidate` objects with
    associated embeddings already attached.
    """
    open_tracks: list[_OpenTrack] = []
    completed: list[TrackCandidate] = []

    for frame in frames:
        # The AI model returns frame_index as the actual timestamp in
        # seconds (the time at which the frame was sampled from the video),
        # NOT a sequential sample counter.
        timestamp_s = frame.frame_index
        # Convert to a sample index for gap detection so that
        # ``max_gap_frames`` counts missed *samples*, not seconds.
        frame_idx = (
            frame.frame_index / frame_interval
            if frame_interval > 0
            else frame.frame_index
        )

        # Gather all face detections across configured categories
        frame_detections: list[Detection] = []
        for cat in detection_categories:
            frame_detections.extend(frame.detections.get(cat, []))

        if not frame_detections:
            # Close tracks that have been unmatched too long
            still_open: list[_OpenTrack] = []
            for trk in open_tracks:
                gap = frame_idx - trk.last_frame_idx
                if gap > max_gap_frames:
                    completed.append(trk.to_candidate())
                else:
                    still_open.append(trk)
            open_tracks = still_open
            continue

        # Greedy matching: for each detection, find best IoU match among open tracks
        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()

        # Build IoU matrix
        pairs: list[tuple[float, int, int]] = []
        for ti, trk in enumerate(open_tracks):
            gap = frame_idx - trk.last_frame_idx
            if gap > max_gap_frames:
                continue  # this track is expired, will be closed below
            for di, det in enumerate(frame_detections):
                iou_val = _iou(trk.last_bbox, det.bbox)
                if iou_val >= iou_threshold:
                    pairs.append((iou_val, ti, di))

        # Sort by IoU descending for greedy assignment
        pairs.sort(key=lambda x: x[0], reverse=True)
        for iou_val, ti, di in pairs:
            if ti in matched_track_indices or di in matched_det_indices:
                continue
            det = frame_detections[di]
            open_tracks[ti].update(
                det.bbox, det.score, timestamp_s, frame_idx, keyframe_iou_threshold,
            )
            matched_track_indices.add(ti)
            matched_det_indices.add(di)

        # Close expired unmatched tracks
        still_open = []
        for ti, trk in enumerate(open_tracks):
            if ti in matched_track_indices:
                still_open.append(trk)
            elif frame_idx - trk.last_frame_idx > max_gap_frames:
                completed.append(trk.to_candidate())
            else:
                still_open.append(trk)

        # Start new tracks for unmatched detections
        for di, det in enumerate(frame_detections):
            if di not in matched_det_indices:
                still_open.append(
                    _OpenTrack(
                        label=label,
                        detector=det.detector,
                        bbox=det.bbox,
                        score=det.score,
                        timestamp_s=timestamp_s,
                        frame_idx=frame_idx,
                    )
                )

        open_tracks = still_open

        # Attach embeddings from this frame's regions to matched tracks
        _attach_frame_embeddings(
            frame, frame_detections, open_tracks, matched_track_indices,
            matched_det_indices, timestamp_s, frame_idx,
        )

    # Close all remaining open tracks
    for trk in open_tracks:
        completed.append(trk.to_candidate())

    return completed


def _attach_frame_embeddings(
    frame: ParsedFrameData,
    frame_detections: list[Detection],
    open_tracks: list[_OpenTrack],
    matched_track_indices: set[int],
    matched_det_indices: set[int],
    timestamp_s: float,
    frame_idx: float,
) -> None:
    """Extract embeddings from frame regions and attach to matching tracks."""
    # Region results are keyed as "regions__<alias>"
    for region_key, region_list in frame.regions.items():
        for region in region_list:
            det_idx = region.detection_index
            if det_idx >= len(frame_detections):
                continue

            emb_keys = [
                k for k, v in region.model_outputs.items()
                if isinstance(v, list) and v and isinstance(v[0], dict) and "vector" in v[0]
            ]
            for emb_cat in emb_keys:
                parsed_embs = parse_embeddings(region, emb_cat)
                for emb in parsed_embs:
                    det = frame_detections[det_idx]
                    fe = FrameEmbedding(
                        vector=emb.vector,
                        norm=emb.norm,
                        score=det.score,
                        bbox=det.bbox,
                        timestamp_s=timestamp_s,
                        embedder=emb.embedder,
                        frame_index=frame_idx,
                    )
                    # Find the open track whose bbox best overlaps this detection
                    best_ti = _find_track_for_detection(open_tracks, det.bbox)
                    if best_ti is not None:
                        open_tracks[best_ti].embeddings.append(fe)


def _find_track_for_detection(
    tracks: list[_OpenTrack],
    det_bbox: list[float],
    threshold: float = 0.50,
) -> int | None:
    """Find the open track whose last_bbox best overlaps with the detection."""
    best_iou = threshold
    best_idx: int | None = None
    for i, trk in enumerate(tracks):
        iou_val = _iou(trk.last_bbox, det_bbox)
        if iou_val > best_iou:
            best_iou = iou_val
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Embedding deduplication  (Phase 2)
# ---------------------------------------------------------------------------

def select_representative_embeddings(
    embeddings: list[FrameEmbedding],
    *,
    max_count: int = DEFAULT_MAX_EMBEDDINGS_PER_TRACK,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> list[FrameEmbedding]:
    """Deduplicate and select representative embeddings for a track.

    1. Sort by quality: ``score * norm`` descending.
    2. Greedily accept: skip if cosine similarity >= *dedup_threshold* to any
       already accepted embedding.
    3. If more than *max_count* survivors, pick top 50% by quality
       and remaining 50% evenly spread across the track's temporal span.
    """
    if not embeddings:
        return []

    # Sort by quality (higher is better)
    scored = sorted(embeddings, key=lambda e: e.score * e.norm, reverse=True)

    # Greedy dedup
    accepted: list[FrameEmbedding] = []
    accepted_vecs: list[np.ndarray] = []

    for emb in scored:
        vec, _ = _l2_normalise(emb.vector)
        too_similar = False
        for av in accepted_vecs:
            sim = float(np.dot(vec, av))
            if sim >= dedup_threshold:
                too_similar = True
                break
        if not too_similar:
            accepted.append(emb)
            accepted_vecs.append(vec)

    if len(accepted) <= max_count:
        return accepted

    # Split: top half by quality, bottom half by temporal spread
    quality_count = max_count // 2
    temporal_count = max_count - quality_count

    by_quality = accepted[:quality_count]
    remaining = accepted[quality_count:]

    # Temporal spread: sort remaining by timestamp, pick evenly spaced
    remaining.sort(key=lambda e: e.timestamp_s or 0.0)
    if len(remaining) <= temporal_count:
        by_temporal = remaining
    else:
        step = len(remaining) / temporal_count
        by_temporal = [remaining[int(i * step)] for i in range(temporal_count)]

    return by_quality + by_temporal


# ---------------------------------------------------------------------------
# Track quality helpers
# ---------------------------------------------------------------------------

def _track_quality(track: TrackCandidate) -> float:
    """Return a quality score for *track* used to sort before cluster matching.

    Higher-quality tracks are processed first so they establish better cluster
    centroids that subsequent lower-quality tracks can then match against.
    """
    if not track.embeddings:
        return 0.0
    return max(e.norm * e.score for e in track.embeddings)


# ---------------------------------------------------------------------------
# Cluster matching  (Phase 3)
# ---------------------------------------------------------------------------

def _match_local_centroids(
    embedding_vec: np.ndarray,
    local_centroids: dict[int, np.ndarray],
    *,
    auto_threshold: float,
    review_threshold: float,
) -> tuple[int | None, float, str]:
    """Match against in-memory centroids built during the current batch.

    Used so that tracks processed later in the same transaction can match
    clusters created by earlier tracks (before the DB commit / centroid
    update happens).
    """
    if not local_centroids:
        return None, 0.0, "new"

    best_cid: int | None = None
    best_sim: float = -1.0
    for cid, centroid in local_centroids.items():
        dot = float(np.dot(embedding_vec, centroid))
        if dot > best_sim:
            best_sim = dot
            best_cid = cid

    if best_cid is not None:
        if best_sim >= auto_threshold:
            return best_cid, best_sim, "auto"
        if best_sim >= review_threshold:
            return best_cid, best_sim, "review"

    return None, best_sim if best_cid is not None else 0.0, "new"


def match_to_cluster(
    embedding_vec: np.ndarray,
    *,
    auto_threshold: float = DEFAULT_AUTO_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    local_centroids: dict[int, np.ndarray] | None = None,
) -> tuple[int | None, float, str]:
    """Match an embedding to the nearest face cluster.

    Checks *local_centroids* first (clusters created earlier in the same
    batch), then falls back to the DB-level pgvector search.

    Returns ``(cluster_id, similarity, match_type)``.
    ``match_type`` is one of: ``"auto"``, ``"review"``, ``"new"``.
    """
    # --- fast path: check in-memory centroids from current batch ---
    if local_centroids:
        cid, sim, mtype = _match_local_centroids(
            embedding_vec, local_centroids,
            auto_threshold=auto_threshold,
            review_threshold=review_threshold,
        )
        if mtype != "new":
            return cid, sim, mtype

    # --- slow path: pgvector ANN on committed clusters ---
    results = find_nearest_cluster(embedding_vec.tolist(), limit=1)
    if not results:
        if local_centroids:
            # Already checked local, no DB match either
            return None, 0.0, "new"
        return None, 0.0, "new"

    cluster_id, similarity = results[0]
    if similarity >= auto_threshold:
        return cluster_id, similarity, "auto"
    elif similarity >= review_threshold:
        return cluster_id, similarity, "review"
    else:
        return None, similarity, "new"


# ---------------------------------------------------------------------------
# Image processing  (main entry point for images)
# ---------------------------------------------------------------------------

async def process_image_detections(
    *,
    run_id: int,
    image_id: int,
    parsed: ParsedImageData,
    classifier: dict[str, str],
    auto_apply_performers: bool = False,
    auto_threshold: float = DEFAULT_AUTO_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    max_exemplars: int = DEFAULT_MAX_EXEMPLARS,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    min_embedding_norm: float = DEFAULT_MIN_EMBEDDING_NORM,
    min_detection_score: float = DEFAULT_MIN_DETECTION_SCORE,
    hard_min_embedding_norm: float = DEFAULT_HARD_MIN_EMBEDDING_NORM,
    hard_min_detection_score: float = DEFAULT_HARD_MIN_DETECTION_SCORE,
    max_embeddings_per_cluster: int = DEFAULT_MAX_EMBEDDINGS_PER_CLUSTER,
) -> dict[str, Any]:
    """Process detections and embeddings for a single image.

    Returns summary dict with counts for logging.
    """
    summary: dict[str, Any] = {
        "tracks_created": 0,
        "embeddings_stored": 0,
        "clusters_matched": 0,
        "clusters_created": 0,
        "performers_applied": [],
        "new_cluster_ids": [],
        "matched_cluster_ids": [],
        "auto_match_cluster_ids": [],
    }

    # Collect face detections
    face_detections: list[Detection] = []
    for cat in _FACE_DETECTION_CATEGORIES:
        face_detections.extend(parsed.detections.get(cat, []))

    if not face_detections:
        # Store non-face detections if present
        await _store_non_face_detections(run_id=run_id, entity_type="image",
                                          entity_id=image_id, parsed_detections=parsed.detections)
        return summary

    # Build a detection-index → embedding mapping from regions
    det_embeddings: dict[int, list[EmbeddingResult]] = {}
    for region_key, region_list in parsed.regions.items():
        for region in region_list:
            emb_keys = [
                k for k, v in region.model_outputs.items()
                if isinstance(v, list) and v and isinstance(v[0], dict) and "vector" in v[0]
            ]
            for emb_cat in emb_keys:
                embs = parse_embeddings(region, emb_cat)
                if embs:
                    det_embeddings.setdefault(region.detection_index, []).extend(embs)

    def _do_image_faces() -> dict[str, Any]:
        # In-memory centroid cache for batch-local matching
        local_centroids: dict[int, np.ndarray] = {}
        local_exemplars: dict[int, list[np.ndarray]] = {}
        cluster_emb_counts: dict[int, int] = {}

        with get_session_local()() as session:
            for det_idx, det in enumerate(face_detections):
                # Create detection track
                track = store_detection_track(
                    session,
                    run_id=run_id,
                    entity_type="image",
                    entity_id=image_id,
                    label=FACE_LABEL,
                    bbox=det.bbox,
                    score=det.score,
                    detector=det.detector,
                )
                summary["tracks_created"] += 1

                # Get embeddings for this detection
                embs = det_embeddings.get(det_idx, [])
                if not embs:
                    continue

                # Two-tier quality filter:
                # 1. Hard floor: reject true junk (false positives, non-faces)
                # 2. Quality floor: gate exemplar eligibility and new cluster creation
                #    Below-quality-but-above-hard faces still get stored and can
                #    match existing clusters, they just won't become exemplars
                #    or seed new clusters.
                hard_pass = (
                    (hard_min_detection_score <= 0 or det.score >= hard_min_detection_score)
                )
                if not hard_pass:
                    _log.debug(
                        "Image %s: skipped face %d (below hard floor: "
                        "score=%.3f, hard_min=%.2f)",
                        image_id, det_idx, det.score, hard_min_detection_score,
                    )
                    continue

                # Filter embeddings by hard norm floor
                embs = [
                    e for e in embs
                    if hard_min_embedding_norm <= 0 or e.norm >= hard_min_embedding_norm
                ]
                if not embs:
                    _log.debug(
                        "Image %s: skipped face %d (all embeddings below hard norm floor %.1f)",
                        image_id, det_idx, hard_min_embedding_norm,
                    )
                    continue

                # Determine if this face meets the quality floor
                meets_quality = (
                    (min_detection_score <= 0 or det.score >= min_detection_score)
                    and any(e.norm >= min_embedding_norm for e in embs)
                )

                # Use best embedding for cluster matching
                best_emb = max(embs, key=lambda e: e.norm)
                vec, norm = _l2_normalise(best_emb.vector)

                # Match to cluster
                cluster_id, similarity, match_type = match_to_cluster(
                    vec,
                    auto_threshold=auto_threshold,
                    review_threshold=review_threshold,
                    local_centroids=local_centroids,
                )

                if match_type == "new":
                    if not meets_quality:
                        # Below quality floor and no existing cluster to join —
                        # don't create a new cluster from a marginal face.
                        _log.debug(
                            "Image %s: below-quality face %d has no cluster match, "
                            "skipping new cluster (score=%.3f, best_norm=%.1f)",
                            image_id, det_idx, det.score, best_emb.norm,
                        )
                        continue
                    # Create new cluster
                    cluster = create_cluster(session, status="unidentified")
                    cluster_id = cluster.id
                    summary["clusters_created"] += 1
                    if cluster_id not in summary["new_cluster_ids"]:
                        summary["new_cluster_ids"].append(cluster_id)
                    _log.debug(
                        "Image %s: new cluster %d for face detection %d",
                        image_id, cluster_id, det_idx,
                    )
                else:
                    summary["clusters_matched"] += 1
                    if cluster_id not in summary["matched_cluster_ids"]:
                        summary["matched_cluster_ids"].append(cluster_id)
                    if match_type == "auto" and cluster_id not in summary["auto_match_cluster_ids"]:
                        summary["auto_match_cluster_ids"].append(cluster_id)
                    _log.debug(
                        "Image %s: matched face %d to cluster %d (sim=%.3f, %s%s)",
                        image_id, det_idx, cluster_id, similarity, match_type,
                        "" if meets_quality else ", below-quality",
                    )

                # Link track to its cluster
                track.cluster_id = cluster_id

                # Store embedding(s), respecting per-cluster cap
                stored_this_cluster = cluster_emb_counts.get(cluster_id, 0)
                remaining_budget = max(
                    0, max_embeddings_per_cluster - stored_this_cluster
                )
                embs_to_store = embs
                if len(embs_to_store) > remaining_budget:
                    embs_to_store = sorted(
                        embs_to_store,
                        key=lambda e: e.score * e.norm,
                        reverse=True,
                    )[:remaining_budget]

                for emb in embs_to_store:
                    emb_vec, emb_norm = _l2_normalise(emb.vector)
                    # Only allow exemplar candidacy if face meets quality floor
                    is_exemplar = False
                    if meets_quality:
                        is_exemplar = try_add_exemplar(
                            session,
                            cluster_id=cluster_id,
                            embedding=emb_vec.tolist(),
                            norm=emb_norm,
                            score=det.score,
                            max_exemplars=max_exemplars,
                            dedup_threshold=dedup_threshold,
                        )
                    store_face_embedding(
                        session,
                        track_id=track.id,
                        cluster_id=cluster_id,
                        entity_type="image",
                        entity_id=image_id,
                        embedding=emb_vec.tolist(),
                        norm=emb_norm,
                        score=det.score,
                        bbox=det.bbox,
                        is_exemplar=is_exemplar,
                        embedder=emb.embedder,
                    )
                    summary["embeddings_stored"] += 1
                    if is_exemplar:
                        local_exemplars.setdefault(cluster_id, []).append(emb_vec)
                    cluster_emb_counts[cluster_id] = cluster_emb_counts.get(cluster_id, 0) + 1

                # Update local centroid for this cluster
                if cluster_id is not None and cluster_id in local_exemplars:
                    vecs = local_exemplars[cluster_id]
                    mean_vec = np.mean(vecs, axis=0)
                    n = np.linalg.norm(mean_vec)
                    if n > 0:
                        mean_vec = mean_vec / n
                    local_centroids[cluster_id] = mean_vec

            session.commit()

            # Update centroids for any affected clusters
            _update_affected_centroids(session, summary)

        # Auto-apply performers for images — only for high-confidence
        # ("auto") matches.  "review"-level matches are surfaced in the
        # FaceReviewPanel for human confirmation.
        if auto_apply_performers:
            applied_performer_ids: set[int] = set()
            auto_cids = set(summary["auto_match_cluster_ids"])
            for cid in auto_cids:
                try:
                    cluster_obj = get_cluster_by_id(cid)
                    if (
                        cluster_obj
                        and cluster_obj.status == "identified"
                        and cluster_obj.performer_id
                        and cluster_obj.performer_id not in applied_performer_ids
                    ):
                        _stash_api.add_performer_to_images([image_id], cluster_obj.performer_id)
                        applied_performer_ids.add(cluster_obj.performer_id)
                        summary["performers_applied"].append(cluster_obj.performer_id)
                        _log.info(
                            "Image %s: auto-applied performer %s from cluster %d",
                            image_id, cluster_obj.performer_id, cid,
                        )
                except Exception:
                    _log.exception(
                        "Image %s: failed to auto-apply performer for cluster %d",
                        image_id, cid,
                    )

        return summary

    result = await asyncio.to_thread(_do_image_faces)

    # Pre-generate face crop thumbnails so FacesHub loads instantly
    thumb_ids = list(set(result.get("new_cluster_ids", []) + result.get("matched_cluster_ids", [])))
    if thumb_ids:
        try:
            from .face_api import precompute_cluster_thumbnails
            cached = await precompute_cluster_thumbnails(thumb_ids)
            _log.debug("Image %s: pre-cached thumbnails for %d cluster(s)", image_id, cached)
        except Exception:
            _log.debug("Thumbnail pre-compute failed for image %s", image_id, exc_info=True)

    # Store non-face detections
    await _store_non_face_detections(
        run_id=run_id, entity_type="image",
        entity_id=image_id, parsed_detections=parsed.detections,
    )

    return result


# ---------------------------------------------------------------------------
# Video processing  (main entry point for scenes)
# ---------------------------------------------------------------------------

async def process_video_detections(
    *,
    run_id: int,
    scene_id: int,
    parsed_frames: list[ParsedFrameData],
    frame_interval: float,
    classifier: dict[str, str],
    auto_apply_performers: bool = False,
    auto_threshold: float = DEFAULT_AUTO_THRESHOLD,
    review_threshold: float = DEFAULT_REVIEW_THRESHOLD,
    max_exemplars: int = DEFAULT_MAX_EXEMPLARS,
    max_embeddings_per_track: int = DEFAULT_MAX_EMBEDDINGS_PER_TRACK,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
    min_embedding_norm: float = DEFAULT_MIN_EMBEDDING_NORM,
    min_detection_score: float = DEFAULT_MIN_DETECTION_SCORE,
    hard_min_embedding_norm: float = DEFAULT_HARD_MIN_EMBEDDING_NORM,
    hard_min_detection_score: float = DEFAULT_HARD_MIN_DETECTION_SCORE,
    max_embeddings_per_cluster: int = DEFAULT_MAX_EMBEDDINGS_PER_CLUSTER,
) -> dict[str, Any]:
    """Process face detections and embeddings for a video/scene.

    Returns summary dict with counts.
    """
    summary: dict[str, Any] = {
        "tracks_created": 0,
        "embeddings_stored": 0,
        "clusters_matched": 0,
        "clusters_created": 0,
        "performers_applied": [],
        "new_cluster_ids": [],
        "matched_cluster_ids": [],
        "auto_match_cluster_ids": [],
    }

    # Phase 1: Build tracks
    tracks = build_tracks(
        parsed_frames,
        frame_interval,
        label=FACE_LABEL,
        detection_categories=_FACE_DETECTION_CATEGORIES,
    )

    if not tracks:
        _log.debug("Scene %s: no face tracks built from %d frames", scene_id, len(parsed_frames))
        return summary

    _log.info(
        "Scene %s: built %d face track(s) from %d frame(s)",
        scene_id, len(tracks), len(parsed_frames),
    )

    # Phase 2: Embedding dedup per track
    for track in tracks:
        if track.embeddings:
            track.embeddings = select_representative_embeddings(
                track.embeddings,
                max_count=max_embeddings_per_track,
                dedup_threshold=dedup_threshold,
            )

    # Phase 2b: Two-tier quality filter.
    # Hard floor: reject true junk (false positives). Only embeddings above
    # the hard floor are kept at all.
    # Quality floor: determines whether a track is eligible to create a new
    # cluster or contribute exemplars. Tracked per-track as a flag.
    before = len(tracks)
    filtered_tracks: list[tuple[TrackCandidate, bool]] = []  # (track, meets_quality)
    hard_dropped = 0
    for t in tracks:
        if not t.embeddings:
            continue
        # Hard floor: reject truly unusable embeddings
        surviving = [
            e for e in t.embeddings
            if (hard_min_detection_score <= 0 or e.score >= hard_min_detection_score)
            and (hard_min_embedding_norm <= 0 or e.norm >= hard_min_embedding_norm)
        ]
        if not surviving:
            hard_dropped += 1
            continue
        t.embeddings = surviving
        # Quality floor: does this track have at least one quality embedding?
        meets_quality = any(
            (min_detection_score <= 0 or e.score >= min_detection_score)
            and (min_embedding_norm <= 0 or e.norm >= min_embedding_norm)
            for e in t.embeddings
        )
        filtered_tracks.append((t, meets_quality))
    if hard_dropped:
        _log.debug(
            "Scene %s: hard-filtered %d of %d track(s) "
            "(hard_min_score=%.2f, hard_min_norm=%.1f)",
            scene_id, hard_dropped, before,
            hard_min_detection_score, hard_min_embedding_norm,
        )
    below_quality = sum(1 for _, mq in filtered_tracks if not mq)
    if below_quality:
        _log.debug(
            "Scene %s: %d track(s) below quality floor (will match existing "
            "clusters only, min_score=%.2f, min_norm=%.1f)",
            scene_id, below_quality, min_detection_score, min_embedding_norm,
        )

    # Sort highest-quality tracks first so they establish cluster centroids
    # before lower-quality tracks are matched.  This dramatically reduces
    # spurious clusters caused by blurry / poorly-aligned frames.
    # Quality tracks first, then below-quality (so they can match clusters
    # that quality tracks created).
    filtered_tracks.sort(
        key=lambda pair: (_track_quality(pair[0]) if pair[1] else 0, _track_quality(pair[0])),
        reverse=True,
    )

    def _do_video_faces() -> dict[str, Any]:
        # In-memory centroid cache so subsequent tracks can match clusters
        # created earlier in this same (uncommitted) batch.
        local_centroids: dict[int, np.ndarray] = {}
        # Accumulate exemplar vectors per cluster for incremental centroid
        local_exemplars: dict[int, list[np.ndarray]] = {}
        # Count how many embeddings we've stored per cluster this run
        # so we can enforce the per-cluster cap.
        cluster_emb_counts: dict[int, int] = {}

        with get_session_local()() as session:
            for track_idx, (track, meets_quality) in enumerate(filtered_tracks):
                # Create detection track row
                db_track = store_detection_track(
                    session,
                    run_id=run_id,
                    entity_type="scene",
                    entity_id=scene_id,
                    label=track.label,
                    bbox=track.best_bbox,
                    score=track.best_score,
                    detector=track.detector,
                    start_s=track.start_s,
                    end_s=track.end_s,
                    keyframes=track.keyframes,
                )
                summary["tracks_created"] += 1

                if not track.embeddings:
                    continue

                # Phase 3: Cluster match using best embedding
                best_emb = max(track.embeddings, key=lambda e: e.norm)
                vec, norm = _l2_normalise(best_emb.vector)

                cluster_id, similarity, match_type = match_to_cluster(
                    vec,
                    auto_threshold=auto_threshold,
                    review_threshold=review_threshold,
                    local_centroids=local_centroids,
                )

                if match_type == "new":
                    if not meets_quality:
                        # Below quality floor and no existing cluster to join.
                        _log.debug(
                            "Scene %s track %d: below-quality, no cluster match, "
                            "skipping (best_score=%.3f, best_norm=%.1f)",
                            scene_id, track_idx, track.best_score,
                            best_emb.norm,
                        )
                        continue
                    cluster = create_cluster(session, status="unidentified")
                    cluster_id = cluster.id
                    summary["clusters_created"] += 1
                    if cluster_id not in summary["new_cluster_ids"]:
                        summary["new_cluster_ids"].append(cluster_id)
                    _log.debug(
                        "Scene %s track %d: new cluster %d",
                        scene_id, track_idx, cluster_id,
                    )
                else:
                    summary["clusters_matched"] += 1
                    if cluster_id not in summary["matched_cluster_ids"]:
                        summary["matched_cluster_ids"].append(cluster_id)
                    if match_type == "auto" and cluster_id not in summary["auto_match_cluster_ids"]:
                        summary["auto_match_cluster_ids"].append(cluster_id)
                    # Seed count from DB on first encounter of existing cluster
                    if cluster_id not in cluster_emb_counts:
                        cluster_emb_counts[cluster_id] = count_cluster_embeddings(cluster_id)
                    _log.debug(
                        "Scene %s track %d: matched to cluster %d (sim=%.3f, %s%s)",
                        scene_id, track_idx, cluster_id, similarity, match_type,
                        "" if meets_quality else ", below-quality",
                    )

                # Link track to its cluster
                db_track.cluster_id = cluster_id

                # Phase 4: Store embeddings (respecting per-cluster cap)
                # All embeddings already passed the hard floor in Phase 2b.
                stored_this_cluster = cluster_emb_counts.get(cluster_id, 0)
                remaining_budget = max(
                    0, max_embeddings_per_cluster - stored_this_cluster
                )
                # Pick the best embeddings to store within budget
                embs_to_store = track.embeddings
                if len(embs_to_store) > remaining_budget:
                    embs_to_store = sorted(
                        embs_to_store,
                        key=lambda e: e.score * e.norm,
                        reverse=True,
                    )[:remaining_budget]

                for fe in embs_to_store:
                    emb_vec, emb_norm = _l2_normalise(fe.vector)
                    # Only allow exemplar candidacy if track meets quality floor
                    is_exemplar = False
                    if meets_quality:
                        is_exemplar = try_add_exemplar(
                            session,
                            cluster_id=cluster_id,
                            embedding=emb_vec.tolist(),
                            norm=emb_norm,
                            score=fe.score,
                            max_exemplars=max_exemplars,
                            dedup_threshold=dedup_threshold,
                        )
                    store_face_embedding(
                        session,
                        track_id=db_track.id,
                        cluster_id=cluster_id,
                        entity_type="scene",
                        entity_id=scene_id,
                        embedding=emb_vec.tolist(),
                        norm=emb_norm,
                        score=fe.score,
                        bbox=fe.bbox,
                        timestamp_s=fe.timestamp_s,
                        is_exemplar=is_exemplar,
                        embedder=fe.embedder,
                    )
                    summary["embeddings_stored"] += 1
                    cluster_emb_counts[cluster_id] = (
                        cluster_emb_counts.get(cluster_id, 0) + 1
                    )
                    # Track exemplar vectors for in-memory centroid
                    if is_exemplar:
                        local_exemplars.setdefault(cluster_id, []).append(emb_vec)

                # Update local centroid so next tracks can match this cluster
                if cluster_id is not None and cluster_id in local_exemplars:
                    vecs = local_exemplars[cluster_id]
                    mean_vec = np.mean(vecs, axis=0)
                    n = np.linalg.norm(mean_vec)
                    if n > 0:
                        mean_vec = mean_vec / n
                    local_centroids[cluster_id] = mean_vec

            session.commit()

            # Update centroids for affected clusters
            _update_affected_centroids(session, summary)

        # Phase 5: Auto-merge — after centroids are committed, merge any
        # clusters produced in this run whose centroids are similar enough to
        # be classified as the same identity.
        all_cluster_ids = list(
            set(summary["new_cluster_ids"]) | set(summary["matched_cluster_ids"])
        )
        merges = _find_auto_merges(all_cluster_ids, auto_threshold=auto_threshold)
        for surviving_id, absorbed_id in merges:
            _log.debug(
                "Scene %s: auto-merging cluster %d into %d",
                scene_id, absorbed_id, surviving_id,
            )
            merge_clusters(surviving_id, absorbed_id)
        if merges:
            summary["clusters_auto_merged"] = len(merges)
            # Remove absorbed cluster IDs from the reported lists so the
            # frontend doesn't try to display merged-away clusters.
            absorbed_ids = {absorbed for _, absorbed in merges}
            summary["new_cluster_ids"] = [
                cid for cid in summary["new_cluster_ids"] if cid not in absorbed_ids
            ]
            summary["matched_cluster_ids"] = [
                cid for cid in summary["matched_cluster_ids"] if cid not in absorbed_ids
            ]

        # Phase 6: Auto-apply performers — only for high-confidence
        # ("auto") matches.  "review"-level matches are surfaced in the
        # FaceReviewPanel for human confirmation rather than tagging
        # automatically.
        if auto_apply_performers:
            applied_performer_ids: set[int] = set()
            auto_cids = set(summary["auto_match_cluster_ids"])
            for cid in auto_cids:
                try:
                    cluster_obj = get_cluster_by_id(cid)
                    if (
                        cluster_obj
                        and cluster_obj.status == "identified"
                        and cluster_obj.performer_id
                        and cluster_obj.performer_id not in applied_performer_ids
                    ):
                        _stash_api.add_performer_to_scenes([scene_id], cluster_obj.performer_id)
                        applied_performer_ids.add(cluster_obj.performer_id)
                        summary["performers_applied"].append(cluster_obj.performer_id)
                        _log.info(
                            "Scene %s: auto-applied performer %s from cluster %d",
                            scene_id, cluster_obj.performer_id, cid,
                        )
                except Exception:
                    _log.exception(
                        "Scene %s: failed to auto-apply performer for cluster %d",
                        scene_id, cid,
                    )

        return summary

    result = await asyncio.to_thread(_do_video_faces)

    # Pre-generate face crop thumbnails so FacesHub loads instantly
    thumb_ids = list(set(result.get("new_cluster_ids", []) + result.get("matched_cluster_ids", [])))
    if thumb_ids:
        try:
            from .face_api import precompute_cluster_thumbnails
            cached = await precompute_cluster_thumbnails(thumb_ids)
            _log.debug("Scene %s: pre-cached thumbnails for %d cluster(s)", scene_id, cached)
        except Exception:
            _log.debug("Thumbnail pre-compute failed for scene %s", scene_id, exc_info=True)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_auto_merges(
    cluster_ids: list[int],
    *,
    auto_threshold: float,
) -> list[tuple[int, int]]:
    """Find clusters from this run that should be auto-merged.

    After centroids have been committed, compare each pair of clusters
    produced during the same processing run.  Pairs whose centroids have
    cosine similarity >= *auto_threshold* are merged (larger cluster
    survives).  Returns ``[(surviving_id, absorbed_id), ...]`` ordered
    highest-similarity first.
    """
    if len(cluster_ids) < 2:
        return []

    import json as _json
    from sqlalchemy import func, select
    from stash_ai_server.models.detections import FaceCluster, FaceEmbedding

    centroids: dict[int, np.ndarray] = {}
    emb_counts: dict[int, int] = {}

    with get_session_local()() as session:
        for cid in cluster_ids:
            row = session.execute(
                select(FaceCluster.centroid).where(FaceCluster.id == cid)
            ).fetchone()
            if row and row[0] is not None:
                raw = row[0]
                if isinstance(raw, str):
                    raw = _json.loads(raw)
                if raw is not None and len(raw) > 0:
                    vec = np.array(raw, dtype=np.float32)
                    n = np.linalg.norm(vec)
                    if n > 0:
                        centroids[cid] = vec / n

        for cid in cluster_ids:
            cnt = session.execute(
                select(func.count())
                .select_from(FaceEmbedding)
                .where(FaceEmbedding.cluster_id == cid)
            ).scalar()
            emb_counts[cid] = cnt or 0

    if len(centroids) < 2:
        return []

    # Enumerate similar pairs
    cid_list = [c for c in cluster_ids if c in centroids]
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(cid_list)):
        for j in range(i + 1, len(cid_list)):
            a, b = cid_list[i], cid_list[j]
            sim = float(np.dot(centroids[a], centroids[b]))
            if sim >= auto_threshold:
                pairs.append((sim, a, b))

    if not pairs:
        return []

    # Greedy merge — highest similarity first, each cluster absorbed at most once
    pairs.sort(reverse=True)
    remaining = set(cid_list)
    merges: list[tuple[int, int]] = []
    for _, a, b in pairs:
        if a not in remaining or b not in remaining:
            continue
        # Larger cluster (by embedding count) survives
        if emb_counts.get(a, 0) >= emb_counts.get(b, 0):
            survivor, absorbed = a, b
        else:
            survivor, absorbed = b, a
        merges.append((survivor, absorbed))
        remaining.discard(absorbed)

    return merges


def _update_affected_centroids(session: Any, summary: dict[str, Any]) -> None:
    """Recompute centroids for all clusters that got new embeddings,
    then check for StashDB matches on newly-created clusters."""
    affected_ids: set[int] = set()
    for cid in summary.get("new_cluster_ids", []):
        affected_ids.add(cid)
    for cid in summary.get("matched_cluster_ids", []):
        affected_ids.add(cid)
    if not affected_ids:
        return
    for cid in affected_ids:
        try:
            update_cluster_centroid(cid)
        except Exception:
            _log.exception("Failed to update centroid for cluster %d", cid)

    # Check StashDB matches for newly-created clusters only — no point
    # re-matching clusters that already have a performer or suggestion.
    new_ids = set(summary.get("new_cluster_ids", []))
    if not new_ids:
        return
    try:
        _check_stashdb_matches_for_clusters(new_ids)
    except Exception:
        _log.debug("StashDB matching skipped or failed", exc_info=True)


def _read_setting_raw(key: str) -> str | None:
    """Read a raw plugin setting value from the DB, returning *None* on any error."""
    try:
        from stash_ai_server.models.plugin import PluginSetting
        from sqlalchemy import select as sa_select
        with get_session_local()() as s:
            row = s.execute(
                sa_select(PluginSetting.value, PluginSetting.default_value)
                .where(
                    PluginSetting.plugin_name == "skier_aitagging",
                    PluginSetting.key == key,
                )
            ).first()
            if row:
                raw = row[0] if row[0] is not None else row[1]
                return raw
    except Exception:
        _log.debug("Could not read setting %s", key, exc_info=True)
    return None


def _read_setting_int(key: str, default: int) -> int:
    """Read an integer plugin setting from the DB, returning *default* on any error."""
    raw = _read_setting_raw(key)
    if raw is not None:
        try:
            return int(float(raw))
        except (ValueError, TypeError):
            pass
    return default


def _read_setting_float(key: str, default: float) -> float:
    """Read a float plugin setting from the DB, returning *default* on any error."""
    raw = _read_setting_raw(key)
    if raw is not None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
    return default


def _check_stashdb_matches_for_clusters(cluster_ids: set[int]) -> None:
    """For each cluster, search StashDB refs and store the best match.

    When the ``stashdb_auto_link_threshold`` plugin setting is > 0 and the
    match similarity exceeds that threshold, the cluster is automatically
    linked to the corresponding local performer (creating one if needed).
    """
    from stash_ai_server.db.stashdb_store import find_nearest_stashdb_ref

    # Quick check: do we even have any imported refs?
    try:
        from stash_ai_server.db.stashdb_store import get_stats
        stats = get_stats()
        if stats["total_refs"] == 0:
            return
    except Exception:
        return

    # Read auto-link threshold from plugin settings (0 = disabled)
    auto_link_threshold = _read_setting_float("stashdb_auto_link_threshold", 0.70)

    for cid in cluster_ids:
        try:
            cluster = get_cluster_by_id(cid)
            if cluster is None or cluster.centroid is None:
                continue
            if cluster.status != "unidentified":
                continue  # already linked
            if getattr(cluster, "stashdb_suggestion_rejected", False):
                continue  # user explicitly marked the previous match as wrong

            centroid = (
                list(cluster.centroid)
                if not isinstance(cluster.centroid, list)
                else cluster.centroid
            )
            matches = find_nearest_stashdb_ref(centroid, limit=1, min_similarity=0.60)
            if not matches:
                continue

            ref_id, stashdb_id, name, similarity = matches[0]
            # Store as suggestion on the cluster
            with get_session_local()() as s:
                from sqlalchemy import update as sa_update
                from stash_ai_server.models.detections import FaceCluster
                s.execute(
                    sa_update(FaceCluster)
                    .where(FaceCluster.id == cid)
                    .values(
                        stashdb_match_id=ref_id,
                        stashdb_match_score=similarity,
                    )
                )
                s.commit()

            _log.debug(
                "Cluster %d matched StashDB performer '%s' (%.1f%%)",
                cid, name, similarity * 100,
            )

            # Auto-link if above threshold
            if auto_link_threshold > 0 and similarity >= auto_link_threshold:
                _auto_link_cluster_to_stashdb(cid, ref_id, stashdb_id, name, similarity)

        except Exception:
            _log.debug("StashDB match check failed for cluster %d", cid, exc_info=True)


def _auto_link_cluster_to_stashdb(
    cluster_id: int,
    ref_id: int,
    stashdb_id: str,
    performer_name: str,
    similarity: float,
) -> None:
    """Auto-link a cluster to the StashDB-matched performer.

    Finds an existing local performer (via stash_id match) or creates a new
    one, then links the cluster and retroactively applies the performer to
    all scenes/images the cluster appears in.
    """
    from stash_ai_server.db.stashdb_store import (
        get_ref_by_id,
        set_local_performer_id,
    )
    from stash_ai_server.db.detection_store import link_performer

    ref = get_ref_by_id(ref_id)
    if ref is None:
        return

    # 1. Check if StashDB ref already has a local performer
    local_pid = ref.local_performer_id

    # 2. If not, look up by stash_id in Stash SQLite
    if local_pid is None:
        try:
            from stash_ai_server.utils import stash_db
            performers_table = stash_db.get_stash_table("performers", required=False)
            stash_ids_table = stash_db.get_stash_table("performer_stash_ids", required=False)
            if performers_table is not None and stash_ids_table is not None:
                import sqlalchemy as sa
                session_factory = stash_db.get_stash_sessionmaker()
                if session_factory is not None:
                    with session_factory() as sess:
                        row = sess.execute(
                            sa.select(
                                performers_table.c.id,
                                performers_table.c.name,
                            )
                            .select_from(
                                performers_table.join(
                                    stash_ids_table,
                                    performers_table.c.id == stash_ids_table.c.performer_id,
                                )
                            )
                            .where(stash_ids_table.c.stash_id == stashdb_id)
                            .limit(1)
                        ).first()
                        if row:
                            local_pid = int(row[0])
        except Exception:
            _log.debug("Could not look up local performer by stash_id", exc_info=True)

    # 3. If still not found, create a new performer in Stash
    #    BUT only if the cluster meets minimum-appearance requirements.
    #    This prevents 300k-scan bulk processing from creating thousands
    #    of unidentified performers for faces seen only once or twice.
    if local_pid is None:
        from stash_ai_server.db.detection_store import get_entity_count_by_type
        counts = get_entity_count_by_type(cluster_id)
        scene_count = counts.get("scene_count", 0)
        image_count = counts.get("image_count", 0)

        min_scenes = _read_setting_int("face_auto_create_min_scenes", 0)
        min_images = _read_setting_int("face_auto_create_min_images", 0)

        # Gate: need at least min_scenes scenes OR min_images images.
        # If both thresholds are 0, always allow (backward-compatible).
        if min_scenes > 0 or min_images > 0:
            if scene_count < min_scenes and image_count < min_images:
                _log.debug(
                    "Auto-link: skipping performer creation for cluster %d "
                    "('%s') — appearances too low (scenes=%d < %d, images=%d < %d)",
                    cluster_id, performer_name,
                    scene_count, min_scenes, image_count, min_images,
                )
                return

        endpoint = ref.source_endpoint or "https://stashdb.org/graphql"
        local_pid = _stash_api.create_performer(
            performer_name,
            stash_ids=[{"stash_id": stashdb_id, "endpoint": endpoint}],
            disambiguation=ref.disambiguation,
        )
        if local_pid is None:
            _log.warning("Auto-link: failed to create performer '%s' for cluster %d", performer_name, cluster_id)
            return
        _log.info("Auto-link: created performer '%s' (id=%d) from StashDB ref", performer_name, local_pid)

    # 4. Update the StashDB ref's local_performer_id
    try:
        set_local_performer_id(ref_id, local_pid)
    except Exception:
        _log.debug("Could not set local_performer_id on ref %d", ref_id, exc_info=True)

    # 5. Link the cluster
    link_performer(cluster_id, local_pid, label=performer_name)

    # 6. Retroactively apply performer to scenes/images
    try:
        from stash_ai_server.db.detection_store import get_cluster_entity_pairs
        entity_pairs = get_cluster_entity_pairs(cluster_id)
        scene_ids = [eid for etype, eid in entity_pairs if etype == "scene"]
        image_ids = [eid for etype, eid in entity_pairs if etype == "image"]
        if scene_ids:
            _stash_api.add_performer_to_scenes(scene_ids, local_pid)
        if image_ids:
            _stash_api.add_performer_to_images(image_ids, local_pid)
    except Exception:
        _log.debug("Auto-link: failed to apply performer to entities for cluster %d", cluster_id, exc_info=True)

    _log.info(
        "Auto-link: cluster %d -> performer '%s' (id=%d) via StashDB match (%.1f%%)",
        cluster_id, performer_name, local_pid, similarity * 100,
    )

    # 7. Hydrate performer from StashDB (set stash_ids + scrape full profile)
    _hydrate_performer_from_stashdb_sync(local_pid, ref)


def _hydrate_performer_from_stashdb_sync(performer_id: int, ref: Any) -> None:
    """Sync hydration for use inside processor threads.

    Mirrors :func:`_hydrate_performer_from_stashdb_ref` in face_api.py but
    uses blocking I/O so it can run in a non-async context.
    """
    try:
        # Step 1: ensure stash_ids link (required for the scraper)
        if getattr(ref, "source_endpoint", None) and getattr(ref, "stashdb_id", None):
            _stash_api.update_performer(
                performer_id,
                stash_ids=[{"stash_id": ref.stashdb_id, "endpoint": ref.source_endpoint}],
            )

        # Step 2: scrape full profile via Stash's built-in stash-box scraper
        if getattr(ref, "source_endpoint", None):
            scraped = _stash_api.scrape_performer_from_stashbox(
                performer_id, ref.source_endpoint
            )
            if scraped:
                _apply_scraped_performer_sync(performer_id, scraped)
                return

        # Step 3: fallback — download image_url stored on the ref
        image_url = getattr(ref, "image_url", None)
        if image_url:
            import urllib.request
            import base64
            try:
                req = urllib.request.Request(image_url, headers={"User-Agent": "StashAIServer/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                    ct = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
                    if not ct or "/" not in ct:
                        ct = "image/jpeg"
                    b64 = base64.b64encode(data).decode("ascii")
                    _stash_api.update_performer_image(performer_id, f"data:{ct};base64,{b64}")
            except Exception:
                _log.debug(
                    "Auto-link: failed to fetch image for performer %d from %s",
                    performer_id, image_url, exc_info=True,
                )
    except Exception:
        _log.debug(
            "Auto-link: hydration failed for performer %d", performer_id, exc_info=True
        )


def _apply_scraped_performer_sync(performer_id: int, scraped: Any) -> None:
    """Normalize and apply stash-box scraped data to a local performer (sync).

    Mirrors the async ``_apply_scraped_performer_data`` in face_api.py.
    """
    import urllib.request
    import base64

    # Unwrap list wrapper that stashapi sometimes returns
    if isinstance(scraped, list):
        scraped = next((s for s in scraped if isinstance(s, dict)), None)
    if isinstance(scraped, dict) and "performer" in scraped:
        scraped = scraped["performer"]
    if not isinstance(scraped, dict):
        return

    update: dict[str, Any] = {"id": performer_id}

    for field in (
        "gender", "birthdate", "death_date", "ethnicity", "country",
        "eye_color", "hair_color", "measurements", "fake_tits",
        "career_length", "tattoos", "piercings", "details",
        "disambiguation", "circumcised",
    ):
        val = scraped.get(field)
        if val:
            update[field] = val

    for int_field in ("height", "height_cm"):
        val = scraped.get(int_field)
        if val:
            try:
                update["height_cm"] = int(val)
                break
            except (ValueError, TypeError):
                pass

    for num_field, key in (("weight", "weight"), ("penis_length", "penis_length")):
        val = scraped.get(num_field)
        if val:
            try:
                update[key] = float(val)
            except (ValueError, TypeError):
                pass

    aliases = scraped.get("aliases") or scraped.get("alias_list")
    if aliases:
        if isinstance(aliases, str):
            update["alias_list"] = [a.strip() for a in aliases.split("\n") if a.strip()]
        elif isinstance(aliases, list):
            alias_list = [a.get("name") if isinstance(a, dict) else a for a in aliases]
            update["alias_list"] = [a for a in alias_list if isinstance(a, str) and a.strip()]

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

    # Download image from stash-box CDN
    images = scraped.get("images") or []
    for img_item in images:
        img_url = img_item.get("url") if isinstance(img_item, dict) else img_item
        if not isinstance(img_url, str) or not img_url.startswith("http"):
            continue
        try:
            req = urllib.request.Request(img_url, headers={"User-Agent": "StashAIServer/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                ct = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
                if not ct or "/" not in ct:
                    ct = "image/jpeg"
                update["image"] = f"data:{ct};base64,{base64.b64encode(data).decode('ascii')}"
            break
        except Exception:
            _log.debug("Auto-link: could not fetch stash-box image from %s", img_url, exc_info=True)

    if "image" not in update:
        img = scraped.get("image")
        if isinstance(img, str) and img.startswith("http"):
            try:
                req = urllib.request.Request(img, headers={"User-Agent": "StashAIServer/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                    ct = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
                    if not ct or "/" not in ct:
                        ct = "image/jpeg"
                    update["image"] = f"data:{ct};base64,{base64.b64encode(data).decode('ascii')}"
            except Exception:
                pass
        elif isinstance(img, str) and img.startswith("data:"):
            update["image"] = img

    if len(update) > 1:
        _stash_api.full_update_performer(update)


async def _store_non_face_detections(
    *,
    run_id: int,
    entity_type: str,
    entity_id: int,
    parsed_detections: dict[str, list[Detection]],
) -> None:
    """Store non-face detections (person, object, etc.) as detection tracks."""
    non_face_dets = {
        cat: dets for cat, dets in parsed_detections.items()
        if cat not in _FACE_DETECTION_CATEGORIES
    }
    if not non_face_dets:
        return

    def _do_store() -> None:
        with get_session_local()() as session:
            for cat, dets in non_face_dets.items():
                for det in dets:
                    store_detection_track(
                        session,
                        run_id=run_id,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        label=cat,
                        bbox=det.bbox,
                        score=det.score,
                        detector=det.detector,
                    )
            session.commit()

    await asyncio.to_thread(_do_store)


def has_embedding_capability(models: Sequence[Any]) -> bool:
    """Check if any active model has detection or embedding capability."""
    for model in models:
        capabilities = getattr(model, "capabilities", [])
        if not capabilities and isinstance(model, dict):
            capabilities = model.get("capabilities", [])
        for cap in capabilities:
            if cap in ("detection", "embedding"):
                return True
    return False
