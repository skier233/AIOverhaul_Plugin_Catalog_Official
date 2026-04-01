from __future__ import annotations

import asyncio
import logging
import time
from typing import Sequence, TypeVar, Awaitable
from stash_ai_server.actions.models import ContextInput
from stash_ai_server.tasks.models import TaskRecord

from .models import AIModelInfo, TagTimeFrame
from .stash_handler import (
    AI_Tagged_Tag_Id,
    add_error_tag_to_images,
    has_ai_tagged,
    has_ai_reprocess,
    is_vr_scene,
    resolve_ai_tag_reference,
    remove_reprocess_tag_from_images,
    remove_reprocess_tag_from_scene,
)
from .http_handler import call_images_api, call_scene_api, get_active_scene_models
from .http_handler import call_face_scan_images_api, call_face_scan_video_api
from .utils import (
    collect_image_tag_records,
    extract_tags_from_response,
    filter_enabled_tag_ids,
    get_selected_items,
)
from .response_parser import (
    build_category_classifier,
    count_detections,
    count_regions,
    parse_embeddings,
    parse_image_result,
    parse_video_frames,
)
from .reprocessing import determine_model_plan
from .marker_handling import apply_scene_markers
from .scene_tagging import apply_scene_tags
from .face_processor import (
    has_embedding_capability,
    process_image_detections,
    process_video_detections,
)
from stash_ai_server.services.base import RemoteServiceBase
from stash_ai_server.tasks.helpers import spawn_chunked_tasks, task_handler
from stash_ai_server.tasks.models import TaskPriority, TaskStatus
from stash_ai_server.tasks.manager import manager as task_manager
from stash_ai_server.utils.stash_api import stash_api
from stash_ai_server.utils.path_mutation import mutate_path_for_plugin
from .legacy_ai_video_result import LegacyAIVideoResult
from .tag_config import get_tag_configuration, resolve_backend_to_stash_tag_id
from stash_ai_server.db.ai_results_store import (
    get_image_model_history_async,
    get_image_tag_ids_async,
    get_scene_model_history_async,
    store_image_run_async,
    store_scene_run_async,
    purge_scene_categories,
)
from stash_ai_server.db.detection_store import cleanup_stale_detections_async
import csv
import logging
from .http_handler import get_active_scene_models

_log = logging.getLogger(__name__)

MAX_IMAGES_PER_REQUEST = 288

SCENE_THRESHOLD = 0.5

current_server_models_cache: list[AIModelInfo] = []

MODELS_CACHE_REFRESH_INTERVAL = 600  # seconds
STASH_CALL_TIMEOUT_SECONDS = 30.0

next_cache_refresh_time = 0.0

T = TypeVar("T")


async def _with_stash_timeout(coro: Awaitable[T], operation: str) -> T:
    try:
        return await asyncio.wait_for(coro, timeout=STASH_CALL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Stash call timed out during {operation}") from exc


def _short_error(message: str, *, limit: int = 120) -> str:
    return message if len(message) <= limit else message[: limit - 3] + "..."


def _format_scene_message(scene_id: int, applied: int, removed: int, markers: int) -> str:
    return (
        f"Scene #{scene_id}: applied {applied} tag(s), "
        f"removed {removed} tag(s), added {markers} marker span(s)."
    )


def _format_multi_summary(kind: str, success: int, failed: int) -> str:
    if failed == 0:
        return f"Processed {success} {kind} successfully."
    return (
        f"{kind.capitalize()} processing finished: {success} succeeded, "
        f"{failed} failed. See AI Tasks for details."
    )


async def _apply_scene_markers_and_tags(
    *,
    scene_id: int,
    service_name: str,
    scene_duration: float,
    existing_scene_tag_ids: Sequence[int] | None,
    apply_ai_tagged_tag: bool = True,
):
    """Reload stored markers and tags for a scene and provide basic counts."""

    markers_by_tag = await apply_scene_markers(
        scene_id=scene_id,
        service_name=service_name,
    )
    tag_changes = await apply_scene_tags(
        scene_id=scene_id,
        service_name=service_name,
        scene_duration=scene_duration,
        existing_scene_tag_ids=existing_scene_tag_ids,
        apply_ai_tagged_tag=apply_ai_tagged_tag,
    )
    marker_count = sum(len(spans) for spans in markers_by_tag.values())
    applied_tags = len(tag_changes.get("applied", []))
    removed_tags = len(tag_changes.get("removed", []))
    return markers_by_tag, tag_changes, marker_count, applied_tags, removed_tags


async def update_model_cache(service: RemoteServiceBase, *, force: bool = False) -> None:
    """Update the cache of models from the remote service."""
    global current_server_models_cache
    global next_cache_refresh_time

    if service.was_disconnected:
        force = True
    now = time.monotonic()
    if not force and now < next_cache_refresh_time:
        return
    try:
        models = await get_active_scene_models(service)
        current_server_models_cache = models
        next_cache_refresh_time = now + MODELS_CACHE_REFRESH_INTERVAL
    except Exception as exc:
        _log.error("Failed to update model cache: %s", exc)

# ==============================================================================
# Image tagging - batch endpoint that accepts multiple image paths
# ==============================================================================


@task_handler(id="skier.ai_tag.image.task")
async def tag_images_task(ctx: ContextInput, params: dict) -> dict:
    """
    Tag images using batch /images endpoint.
    """
    _log.info("Starting image tagging task for context: %s", ctx)
    raw_image_ids = get_selected_items(ctx)
    service: RemoteServiceBase = params["service"]
    apply_ai_tagged_tag = service.apply_ai_tagged_tag

    image_ids: list[int] = []
    for raw in raw_image_ids:
        try:
            image_ids.append(int(raw))
        except (TypeError, ValueError):
            _log.warning("Skipping invalid image id: %s", raw)

    if not image_ids:
        return {
            "status": "noop",
            "message": "No images to process.",
            "processed_ids": [],
            "failed_ids": [],
            "skipped_ids": [],
            "tags_added": {},
        }

    try:
        image_metadata = await _with_stash_timeout(
            stash_api.get_image_paths_and_tags_async(image_ids),
            "get_image_paths_and_tags",
        )
    except Exception as exc:
        _log.exception("Failed to fetch image metadata for ids=%s", image_ids)
        detail = _short_error(str(exc) or exc.__class__.__name__)
        return {
            "status": "failed",
            "message": f"Image tagging failed while fetching paths ({detail}).",
            "processed_ids": image_ids,
            "failed_ids": image_ids,
            "skipped_ids": [],
            "tags_added": {},
        }

    failure_reasons: dict[int, str] = {}
    failed_images: set[int] = set()
    skipped_images: set[int] = set()

    # Face summary aggregation across all images
    _agg_faces_new = 0
    _agg_faces_matched = 0
    _agg_faces_total = 0
    _agg_new_cluster_ids: list[int] = []
    _agg_matched_cluster_ids: list[int] = []

    valid_paths: dict[int, str] = {}
    reprocess_request_ids: set[int] = set()
    for image_id in image_ids:
        record = (image_metadata or {}).get(image_id) or {}
        path = record.get("path") if isinstance(record, dict) else None
        tags = record.get("tag_ids") if isinstance(record, dict) else None
        print("Image ID %s has tags: %s" % (image_id, tags))
        if has_ai_reprocess(tags):
            print("Image ID %s has AI_Reprocess tag; scheduling reprocess" % image_id)
            reprocess_request_ids.add(image_id)
        if not path:
            failure_reasons[image_id] = "file path unavailable"
            failed_images.add(image_id)
            continue
        valid_paths[image_id] = path

    if not valid_paths:
        return {
            "status": "failed",
            "message": "No valid images to process.",
            "processed_ids": image_ids,
            "failed_ids": image_ids,
            "skipped_ids": [],
            "tags_added": {},
        }

    await update_model_cache(service)

    config = get_tag_configuration()
    active_models = list(current_server_models_cache)
    requested_models_payload = [model.model_dump(exclude_none=True) for model in active_models]

    remote_targets: dict[int, str] = {}
    determine_errors: list[int] = []

    for image_id, path in valid_paths.items():
        try:
            historical_models = await get_image_model_history_async(service=service.name, image_id=image_id)
        except Exception:
            _log.exception("Failed to load image model history for image_id=%s", image_id)
            historical_models = ()
            determine_errors.append(image_id)

        _, should_reprocess = determine_model_plan(
            current_models=active_models,
            previous_models=historical_models,
            current_frame_interval=service.tagging_frame_interval if hasattr(service, "tagging_frame_interval") else 2.0,
            current_threshold=SCENE_THRESHOLD,
        )

        if image_id in reprocess_request_ids:
            _log.info("AI_Reprocess tag present for image_id=%s; forcing reprocess", image_id)
            should_reprocess = True

        if should_reprocess:
            remote_targets[image_id] = mutate_path_for_plugin(path, service.plugin_name)
        else:
            skipped_images.add(image_id)

    if remote_targets:
        remote_image_ids = list(remote_targets.keys())
        remote_paths = [remote_targets[iid] for iid in remote_image_ids]
        try:
            response = await call_images_api(service, remote_paths)
            _log.debug("Images API metrics: %s", getattr(response, "metrics", None))
        except Exception:
            _log.exception("Remote image tagging failed for %d images", len(remote_image_ids))
            add_error_tag_to_images(remote_image_ids)
            for iid in remote_image_ids:
                failure_reasons[iid] = "remote service request failed"
            failed_images.update(remote_image_ids)
            response = None

        if response is not None:
            result_payload = response.result if isinstance(response.result, list) else []
            models_used = response.models if getattr(response, "models", None) else []
            # Serialise to dicts — _upsert_models expects Mapping objects
            models_used_payload = [
                m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
                for m in models_used
            ] if models_used else []
            classifier = build_category_classifier(models_used)
            for idx, image_id in enumerate(remote_image_ids):
                payload = result_payload[idx] if idx < len(result_payload) else {}

                parsed_image = parse_image_result(
                    payload if isinstance(payload, dict) else {}, classifier
                )
                if parsed_image.error:
                    failure_reasons[image_id] = _short_error(parsed_image.error)
                    add_error_tag_to_images([image_id])
                    failed_images.add(image_id)
                    continue

                # Existing tag processing (unchanged)
                tags_by_category = parsed_image.tags
                resolved_records = collect_image_tag_records(tags_by_category, config)

                # Log structured data so we can verify the v3 response is parsed correctly
                det_count = count_detections(parsed_image)
                region_count = count_regions(parsed_image)
                _log.debug(
                    "Image %s parsed: %d tag category(ies) %s, "
                    "%d detection(s) across %s, %d region group(s) %s",
                    image_id,
                    len(parsed_image.tags),
                    list(parsed_image.tags.keys()),
                    det_count,
                    list(parsed_image.detections.keys()),
                    region_count,
                    list(parsed_image.regions.keys()),
                )
                for det_cat, dets in parsed_image.detections.items():
                    for i, det in enumerate(dets):
                        _log.debug(
                            "  Image %s detection [%s][%d]: bbox=%s score=%.3f detector=%s",
                            image_id, det_cat, i, det.bbox, det.score, det.detector,
                        )
                for region_key, region_list in parsed_image.regions.items():
                    for reg in region_list:
                        emb_keys = [
                            k for k, v in reg.model_outputs.items()
                            if isinstance(v, list) and v and isinstance(v[0], dict) and "vector" in v[0]
                        ]
                        for emb_cat in emb_keys:
                            embeddings = parse_embeddings(reg, emb_cat)
                            for emb in embeddings:
                                _log.debug(
                                    "  Image %s region [%s] det_idx=%d: %s embedder=%s norm=%.2f dim=%d",
                                    image_id, region_key, reg.detection_index,
                                    emb_cat, emb.embedder, emb.norm, len(emb.vector),
                                )
                run_id: int | None = None
                try:
                    run_id = await store_image_run_async(
                        service=service.name,
                        plugin_name=service.plugin_name,
                        image_id=image_id,
                        tag_records=resolved_records,
                        input_params=None,
                        requested_models=models_used_payload if models_used_payload else requested_models_payload,
                    )
                except Exception:
                    _log.exception("Failed to persist image tagging run for image_id=%s", image_id)

                # --- Face / detection processing (non-blocking) ---
                if run_id is not None and (parsed_image.detections or parsed_image.regions) and has_embedding_capability(models_used):
                    # Clean up detection tracks from prior runs for this image
                    try:
                        stale = await cleanup_stale_detections_async(
                            entity_type="image",
                            entity_id=image_id,
                            service=service.name,
                            exclude_run_id=run_id,
                        )
                        if stale:
                            _log.debug("Image %s: purged %d stale detection track(s)", image_id, stale)
                    except Exception:
                        _log.exception("Failed to cleanup stale detections for image %s", image_id)
                    try:
                        face_summary = await process_image_detections(
                            run_id=run_id,
                            image_id=image_id,
                            parsed=parsed_image,
                            classifier=classifier,
                            auto_apply_performers=service.auto_apply_performers,
                            auto_threshold=service.face_match_auto_threshold,
                            review_threshold=service.face_match_review_threshold,
                            max_exemplars=service.face_max_exemplars_per_cluster,
                            dedup_threshold=service.face_embedding_dedup_threshold,
                            min_embedding_norm=service.face_min_embedding_norm,
                            min_detection_score=service.face_min_detection_score,
                            hard_min_embedding_norm=service.face_hard_min_embedding_norm,
                            hard_min_detection_score=service.face_hard_min_detection_score,
                            max_embeddings_per_cluster=service.face_max_embeddings_per_cluster,
                        )
                        if face_summary.get("tracks_created"):
                            _log.debug(
                                "Image %s face processing: %d track(s), %d embedding(s), "
                                "%d cluster match(es), %d new cluster(s)",
                                image_id,
                                face_summary["tracks_created"],
                                face_summary["embeddings_stored"],
                                face_summary["clusters_matched"],
                                face_summary["clusters_created"],
                            )
                            _agg_faces_new += face_summary["clusters_created"]
                            _agg_faces_matched += face_summary["clusters_matched"]
                            _agg_faces_total += face_summary["tracks_created"]
                            for cid in face_summary.get("new_cluster_ids", []):
                                if cid not in _agg_new_cluster_ids:
                                    _agg_new_cluster_ids.append(cid)
                            for cid in face_summary.get("matched_cluster_ids", []):
                                if cid not in _agg_matched_cluster_ids:
                                    _agg_matched_cluster_ids.append(cid)
                    except Exception:
                        _log.exception("Failed to process detections for image %s", image_id)

    tags_added_counts: dict[int, int] = {}

    for image_id in image_ids:
        tags_added_counts[image_id] = 0
        if image_id in failed_images:
            continue
        try:
            stored_tag_ids = await get_image_tag_ids_async(service=service.name, image_id=image_id)
        except Exception:
            _log.exception("Failed to load stored image tags for image_id=%s", image_id)
            failure_reasons[image_id] = "failed to load stored tags"
            failed_images.add(image_id)
            continue

        normalized_ids = filter_enabled_tag_ids(stored_tag_ids, config)
        if apply_ai_tagged_tag and AI_Tagged_Tag_Id:
            normalized_ids = list(dict.fromkeys([*normalized_ids, AI_Tagged_Tag_Id]))
        tags_added_counts[image_id] = len(normalized_ids)

        if not normalized_ids and not stored_tag_ids:
            continue

        try:
            if stored_tag_ids:
                await _with_stash_timeout(
                    stash_api.remove_tags_from_images_async([image_id], stored_tag_ids),
                    "remove_tags_from_images",
                )
            if normalized_ids:
                await _with_stash_timeout(
                    stash_api.add_tags_to_images_async([image_id], normalized_ids),
                    "add_tags_to_images",
                )
        except Exception:
            _log.exception("Failed to refresh tags for image_id=%s", image_id)
            failure_reasons[image_id] = "failed to sync tags with Stash"
            failed_images.add(image_id)

    processed_ids = list(dict.fromkeys(image_ids))
    failed_ids = sorted(failed_images)
    skipped_ids = sorted(skipped_images)
    success_count = len(processed_ids) - len(failed_ids)

    reprocess_cleared = [
        image_id
        for image_id in processed_ids
        if image_id in reprocess_request_ids and image_id not in failed_images
    ]
    if reprocess_cleared:
        await _with_stash_timeout(
            remove_reprocess_tag_from_images(reprocess_cleared),
            "remove_reprocess_tag_from_images",
        )

    status = "success"
    if failed_ids:
        status = "failed" if success_count == 0 else "partial"

    if len(processed_ids) == 1:
        image_id = processed_ids[0]
        if image_id in failed_images:
            reason = failure_reasons.get(image_id, "unknown error")
            message = f"Image #{image_id}: tagging failed ({reason})."
        else:
            added = tags_added_counts.get(image_id, 0)
            message = f"Image #{image_id}: added {added} tag(s)."
    else:
        message = _format_multi_summary("images", success_count, len(failed_ids))

    return {
        "status": status,
        "message": message,
        "processed_ids": processed_ids,
        "failed_ids": failed_ids,
        "skipped_ids": skipped_ids,
        "tags_added": tags_added_counts,
        "failure_reasons": {iid: failure_reasons[iid] for iid in failed_ids},
        "face_summary": {
            "faces_new": _agg_faces_new,
            "faces_matched": _agg_faces_matched,
            "faces_total": _agg_faces_total,
            "new_cluster_ids": _agg_new_cluster_ids,
            "matched_cluster_ids": _agg_matched_cluster_ids,
        } if (_agg_faces_new or _agg_faces_matched) else None,
    }


# ==============================================================================
# Scene tagging
# ==============================================================================


@task_handler(id="skier.ai_tag.scene.task")
async def tag_scene_task(ctx: ContextInput, params: dict, task_record: TaskRecord) -> dict:
    scene_id_raw = ctx.entity_id
    _log.debug("ASYNC debug scene id: %s", scene_id_raw)
    if scene_id_raw is None:
        raise ValueError("Context missing scene entity_id. ctx: %s" % ctx)
    try:
        scene_id = int(scene_id_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid scene_id: {scene_id_raw}") from exc

    service = params["service"]
    try:
        scene_path, scene_tags, scene_duration = await _with_stash_timeout(
            stash_api.get_scene_path_and_tags_and_duration_async(scene_id),
            "get_scene_path_and_tags_and_duration",
        )
    except Exception as exc:
        _log.exception("Failed to load scene metadata for scene_id=%s", scene_id)
        detail = _short_error(str(exc) or exc.__class__.__name__)
        return {
            "scene_id": scene_id,
            "status": "failed",
            "message": f"Scene #{scene_id}: tagging failed ({detail}).",
            "processed_ids": [scene_id],
            "failed_ids": [scene_id],
        }

    remote_scene_path = mutate_path_for_plugin(scene_path or "", service.plugin_name)
    force_reprocess = has_ai_reprocess(scene_tags)

    try:
        historical_models = await get_scene_model_history_async(service=service.name, scene_id=scene_id)
    except Exception:
        _log.exception("Failed to load scene model history for scene_id=%s", scene_id)
        historical_models = ()
    if has_ai_tagged(scene_tags) and not historical_models:
        legacy_result = await LegacyAIVideoResult.try_load_from_scene_path(scene_path)
        if legacy_result is None:
            _log.debug("No legacy AI json found for scene_id=%s", scene_id)
        else:
            imported = await legacy_result.save_to_db(scene_id=scene_id, service=service)
            if imported:
                historical_models = await get_scene_model_history_async(service=service.name, scene_id=scene_id)

    if historical_models:
        _log.debug(
            "Scene %s historical models: %s",
            scene_id,
            [m.model_name for m in historical_models],
        )

    await update_model_cache(service)

    skip_categories, should_reprocess = determine_model_plan(
        current_models=current_server_models_cache,
        previous_models=historical_models,
        current_frame_interval=service.tagging_frame_interval if hasattr(service, "tagging_frame_interval") else 2.0,
        current_threshold=SCENE_THRESHOLD,
    )

    if force_reprocess:
        _log.info("AI_Reprocess tag present for scene_id=%s; forcing full reprocess", scene_id)
        skip_categories = tuple()
        should_reprocess = True

    try:
        if not should_reprocess:
            _log.debug("Skipping remote tagging for scene_id=%s; existing data considered current", scene_id)
            (
                markers_by_tag,
                tag_changes,
                marker_count,
                applied_tags,
                removed_tags,
            ) = await _apply_scene_markers_and_tags(
                scene_id=scene_id,
                service_name=service.name,
                scene_duration=scene_duration,
                existing_scene_tag_ids=scene_tags,
                apply_ai_tagged_tag=service.apply_ai_tagged_tag,
            )
            message = _format_scene_message(scene_id, applied_tags, removed_tags, marker_count)
            summary_parts = [f"Retrieved {marker_count} marker span(s) from storage"]
            if applied_tags:
                summary_parts.append(f"applied {applied_tags} scene tag(s)")
            if removed_tags:
                summary_parts.append(f"removed {removed_tags} scene tag(s)")
            return {
                "scene_id": scene_id,
                "status": "success",
                "message": message,
                "scene_tags": tag_changes,
                "summary": "; ".join(summary_parts),
                "markers_applied": marker_count,
                "tags_applied": applied_tags,
                "tags_removed": removed_tags,
                "processed_ids": [scene_id],
                "failed_ids": [],
            }

        vr_scene = is_vr_scene(scene_tags)
        _log.debug(
            "Running scene tagging for scene_id=%s; skipping categories=%s",
            scene_id,
            skip_categories,
        )
        response = await call_scene_api(
            service,
            remote_scene_path,
            service.tagging_frame_interval if hasattr(service, "tagging_frame_interval") else 2.0,
            vr_scene,
            threshold=SCENE_THRESHOLD,
            skip_categories=skip_categories,
        )
        if response is not None:
            _log.debug("Scene API metrics: %s", response.metrics)

        if response is None or response.result is None:
            _log.warning("Remote scene tagging returned no data for scene_id=%s", scene_id)
            (
                markers_by_tag,
                tag_changes,
                marker_count,
                applied_tags,
                removed_tags,
            ) = await _apply_scene_markers_and_tags(
                scene_id=scene_id,
                service_name=service.name,
                scene_duration=scene_duration,
                existing_scene_tag_ids=scene_tags,
                apply_ai_tagged_tag=service.apply_ai_tagged_tag,
            )
            message = (
                f"Scene #{scene_id}: remote service returned no data. "
                f"Applied {applied_tags} tag(s), removed {removed_tags}, added {marker_count} marker span(s) from storage."
            )
            summary_parts = ["Remote tagging service returned no data"]
            if marker_count:
                summary_parts.append(f"reapplied {marker_count} marker span(s) from storage")
            if applied_tags:
                summary_parts.append(f"applied {applied_tags} scene tag(s)")
            if removed_tags:
                summary_parts.append(f"removed {removed_tags} scene tag(s)")
            return {
                "scene_id": scene_id,
                "status": "failed",
                "message": message,
                "scene_tags": tag_changes,
                "summary": "; ".join(summary_parts),
                "markers_applied": marker_count,
                "tags_applied": applied_tags,
                "tags_removed": removed_tags,
                "processed_ids": [scene_id],
                "failed_ids": [scene_id],
            }

        result = response.result
        processed_categories = {
            str(category)
            for category in (result.timespans.keys() if result.timespans else [])
            if category is not None
        }

        # Parse structured per-frame data (detections, regions, embeddings)
        parsed_frames = None
        if result.frames:
            frame_classifier = build_category_classifier(result.models)
            parsed_frames = parse_video_frames(result.frames, frame_classifier)
            if parsed_frames:
                total_dets = sum(count_detections(f) for f in parsed_frames)
                total_regions = sum(count_regions(f) for f in parsed_frames)
                det_cats = set()
                region_keys = set()
                for f in parsed_frames:
                    det_cats.update(f.detections.keys())
                    region_keys.update(f.regions.keys())
                _log.debug(
                    "Scene %s: %d frame(s) with structured data, "
                    "%d total detection(s) across %s, "
                    "%d total region result(s) across %s",
                    scene_id, len(parsed_frames),
                    total_dets, sorted(det_cats),
                    total_regions, sorted(region_keys),
                )
                # Log per-frame detail for the first few frames to aid verification
                frames_to_log = parsed_frames[:5]
                for pf in frames_to_log:
                    for det_cat, dets in pf.detections.items():
                        for i, det in enumerate(dets):
                            _log.debug(
                                "  Scene %s frame %.1f detection [%s][%d]: "
                                "bbox=%s score=%.3f detector=%s",
                                scene_id, pf.frame_index, det_cat, i,
                                det.bbox, det.score, det.detector,
                            )
                    for region_key, region_list in pf.regions.items():
                        for reg in region_list:
                            emb_keys = [
                                k for k, v in reg.model_outputs.items()
                                if isinstance(v, list) and v and isinstance(v[0], dict) and "vector" in v[0]
                            ]
                            for emb_cat in emb_keys:
                                embeddings = parse_embeddings(reg, emb_cat)
                                for emb in embeddings:
                                    _log.debug(
                                        "  Scene %s frame %.1f region [%s] det_idx=%d: "
                                        "%s embedder=%s norm=%.2f dim=%d",
                                        scene_id, pf.frame_index, region_key, reg.detection_index,
                                        emb_cat, emb.embedder, emb.norm, len(emb.vector),
                                    )
                if len(parsed_frames) > 5:
                    _log.debug(
                        "  Scene %s: ... and %d more frame(s) with structured data",
                        scene_id, len(parsed_frames) - 5,
                    )

        result_models_payload = [model.model_dump(exclude_none=True) for model in result.models]

        missing_from_cache = [m for m in result.models if m not in current_server_models_cache]
        if missing_from_cache:
            _log.debug(
                "Discovered %d models not present in cache; triggering refresh", len(missing_from_cache)
            )
            await update_model_cache(service, force=True)

        run_id: int | None = None
        try:
            tag_config = get_tag_configuration()

            run_id = await store_scene_run_async(
                service=service.name,
                plugin_name=service.plugin_name,
                scene_id=scene_id,
                input_params={
                    "frame_interval": service.tagging_frame_interval if hasattr(service, "tagging_frame_interval") else 2.0,
                    "vr_video": vr_scene,
                    "threshold": SCENE_THRESHOLD,
                },
                result_payload=result.model_dump(exclude_none=True),
                requested_models=result_models_payload,
                resolve_reference=lambda backend_label, category: resolve_backend_to_stash_tag_id(backend_label, tag_config, category),
            )
        except Exception:
            _log.exception("Failed to persist AI scene run for scene_id=%s", scene_id)

        if run_id is not None and processed_categories:
            purge_scene_categories(
                service=service.name,
                scene_id=scene_id,
                categories=processed_categories,
                exclude_run_id=run_id,
            )

        # --- Face / detection processing (non-blocking) ---
        face_summary_parts: list[str] = []
        face_result_data: dict | None = None
        if parsed_frames and run_id is not None and has_embedding_capability(result.models):
            # Clean up detection tracks from prior runs for this scene
            # so reprocessing replaces data instead of duplicating it.
            try:
                stale = await cleanup_stale_detections_async(
                    entity_type="scene",
                    entity_id=scene_id,
                    service=service.name,
                    exclude_run_id=run_id,
                )
                if stale:
                    _log.debug("Scene %s: purged %d stale detection track(s)", scene_id, stale)
            except Exception:
                _log.exception("Failed to cleanup stale detections for scene %s", scene_id)
            try:
                frame_classifier = build_category_classifier(result.models)
                face_summary = await process_video_detections(
                    run_id=run_id,
                    scene_id=scene_id,
                    parsed_frames=parsed_frames,
                    frame_interval=result.frame_interval,
                    classifier=frame_classifier,
                    auto_apply_performers=service.auto_apply_performers,
                    auto_threshold=service.face_match_auto_threshold,
                    review_threshold=service.face_match_review_threshold,
                    max_exemplars=service.face_max_exemplars_per_cluster,
                    max_embeddings_per_track=service.face_max_embeddings_per_track,
                    dedup_threshold=service.face_embedding_dedup_threshold,
                    min_embedding_norm=service.face_min_embedding_norm,
                    min_detection_score=service.face_min_detection_score,
                    hard_min_embedding_norm=service.face_hard_min_embedding_norm,
                    hard_min_detection_score=service.face_hard_min_detection_score,
                    max_embeddings_per_cluster=service.face_max_embeddings_per_cluster,
                )
                if face_summary.get("tracks_created"):
                    auto_merged = face_summary.get("clusters_auto_merged", 0)
                    merge_part = f", {auto_merged} auto-merged" if auto_merged else ""
                    face_msg = (
                        f"Faces: {face_summary['tracks_created']} detected, "
                        f"{face_summary['clusters_matched']} matched, "
                        f"{face_summary['clusters_created']} new cluster(s){merge_part}"
                    )
                    face_summary_parts.append(face_msg)
                    _log.debug("Scene %s face processing: %s", scene_id, face_msg)
                    face_result_data = {
                        "faces_new": len(face_summary.get("new_cluster_ids", [])),
                        "faces_matched": face_summary["clusters_matched"],
                        "faces_total": face_summary["tracks_created"],
                        "faces_auto_merged": auto_merged,
                        "new_cluster_ids": face_summary.get("new_cluster_ids", []),
                        "matched_cluster_ids": face_summary.get("matched_cluster_ids", []),
                    }
            except Exception:
                _log.exception("Failed to process detections for scene %s", scene_id)

        (
            markers_by_tag,
            tag_changes,
            marker_count,
            applied_tags,
            removed_tags,
        ) = await _apply_scene_markers_and_tags(
            scene_id=scene_id,
            service_name=service.name,
            scene_duration=scene_duration,
            existing_scene_tag_ids=scene_tags,
            apply_ai_tagged_tag=service.apply_ai_tagged_tag,
        )
        message = _format_scene_message(scene_id, applied_tags, removed_tags, marker_count)
        summary_parts = [f"Processed scene with {marker_count} marker span(s)"]
        if applied_tags:
            summary_parts.append(f"applied {applied_tags} scene tag(s)")
        if removed_tags:
            summary_parts.append(f"removed {removed_tags} scene tag(s)")
        summary_parts.extend(face_summary_parts)

        if force_reprocess:
            await remove_reprocess_tag_from_scene(scene_id)

        return {
            "scene_id": scene_id,
            "status": "success",
            "message": message,
            "scene_tags": tag_changes,
            "summary": "; ".join(summary_parts),
            "markers_applied": marker_count,
            "tags_applied": applied_tags,
            "tags_removed": removed_tags,
            "processed_ids": [scene_id],
            "failed_ids": [],
            "face_summary": face_result_data,
        }
    except Exception as exc:
        _log.exception("Scene tagging failed for scene_id=%s", scene_id)
        detail = _short_error(str(exc) or exc.__class__.__name__)
        return {
            "scene_id": scene_id,
            "status": "failed",
            "message": f"Scene #{scene_id}: tagging failed ({detail}).",
            "processed_ids": [scene_id],
            "failed_ids": [scene_id],
        }


async def tag_scenes(service: RemoteServiceBase, ctx: ContextInput, params: dict, task_record: TaskRecord):
    selected_items = get_selected_items(ctx)
    params["service"] = service
    if not selected_items:
        return {
            "status": "noop",
            "message": "No scenes to process.",
            "scenes_requested": 0,
            "scenes_completed": 0,
            "scenes_failed": 0,
        }
    if len(selected_items) == 1:
        if not ctx.entity_id:
            ctx.entity_id = str(selected_items[0])
        result = await tag_scene_task(ctx, params, task_record)
        return result

    task_priority = TaskPriority.low
    if ctx.is_detail_view:
        task_priority = TaskPriority.high
    elif ctx.selected_ids and len(ctx.selected_ids) >= 1:
        task_priority = TaskPriority.normal
    elif ctx.visible_ids and len(ctx.visible_ids) >= 1:
        task_priority = TaskPriority.normal

    spawn_result = await spawn_chunked_tasks(
        parent_task=task_record,
        parent_context=ctx,
        handler=tag_scene_task,
        items=selected_items,
        chunk_size=1,
        params=params,
        priority=task_priority,
        hold_children=True,
    )
    child_ids = spawn_result.get("spawned", [])
    success = 0
    failed = 0
    for child_id in child_ids:
        child = task_manager.get(child_id)
        if child is None:
            continue
        if child.status == TaskStatus.failed:
            failed += 1
            continue
        child_result = getattr(child, "result", None)
        if isinstance(child_result, dict) and child_result.get("status") == "failed":
            failed += 1
        else:
            success += 1

    total_requested = len(selected_items)
    accounted = success + failed
    if accounted < len(child_ids):
        failed += len(child_ids) - accounted
        accounted = success + failed
    if accounted < total_requested:
        # Treat any unaccounted requested scenes as failures to be safe.
        failed += total_requested - accounted

    failed = min(failed, total_requested)
    success = max(total_requested - failed, 0)

    status = "success"
    if failed:
        status = "failed" if success == 0 else "partial"
    message = _format_multi_summary("scenes", success, failed)

    return {
        "status": status,
        "message": message,
        "scenes_requested": total_requested,
        "scenes_completed": success,
        "scenes_failed": failed,
        "spawned": child_ids,
        "count": len(child_ids),
        "held": spawn_result.get("held", False),
    }

async def tag_images(service: RemoteServiceBase, ctx: ContextInput, params: dict, task_record: TaskRecord):
    selected_items = get_selected_items(ctx)
    params["service"] = service
    if not selected_items:
        return {
            "status": "noop",
            "message": "No images to process.",
            "images_requested": 0,
            "images_completed": 0,
            "images_failed": 0,
        }
    if len(selected_items) <= MAX_IMAGES_PER_REQUEST:
        return await tag_images_task(ctx, params)

    task_priority = TaskPriority.low
    if ctx.is_detail_view:
        task_priority = TaskPriority.high
    elif ctx.selected_ids and len(ctx.selected_ids) >= 1:
        task_priority = TaskPriority.normal
    elif ctx.visible_ids and len(ctx.visible_ids) >= 1:
        task_priority = TaskPriority.normal

    spawn_result = await spawn_chunked_tasks(
        parent_task=task_record,
        parent_context=ctx,
        handler=tag_images_task,
        items=selected_items,
        chunk_size=MAX_IMAGES_PER_REQUEST,
        params=params,
        priority=task_priority,
        hold_children=True,
    )
    child_ids = spawn_result.get("spawned", [])
    total_requested = len(selected_items)
    success = 0
    failed = 0

    for child_id in child_ids:
        child = task_manager.get(child_id)
        if child is None:
            continue

        chunk_total = 0
        try:
            if child.context.selected_ids:
                chunk_total = len(child.context.selected_ids)
            elif child.context.entity_id:
                chunk_total = 1
        except Exception:
            chunk_total = 0

        if child.status == TaskStatus.failed:
            failed += chunk_total or 1
            continue

        child_result = getattr(child, "result", None)
        if isinstance(child_result, dict):
            processed = child_result.get("processed_ids") or []
            failed_ids = child_result.get("failed_ids") or []
            success += max(len(processed) - len(failed_ids), 0)
            failed += len(failed_ids)
        else:
            failed += chunk_total or 1

    failed = min(failed, total_requested)
    success = max(total_requested - failed, 0)

    status = "success"
    if failed:
        status = "failed" if success == 0 else "partial"
    message = _format_multi_summary("images", success, failed)

    return {
        "status": status,
        "message": message,
        "images_requested": total_requested,
        "images_completed": success,
        "images_failed": failed,
        "spawned": child_ids,
        "count": len(child_ids),
        "held": spawn_result.get("held", False),
    }


# ==============================================================================
# Face-scan-only (no tag classification)
# ==============================================================================

MAX_FACE_SCAN_IMAGES_PER_REQUEST = 288

# Threshold sentinel for face-scan runs — stored in model history so that
# reprocessing comparisons between face-scan runs are self-consistent, and
# switching between tagging ↔ face-scan correctly triggers reprocess.
FACE_SCAN_THRESHOLD = 0.0


def _filter_embedding_models(models: list[AIModelInfo]) -> list[AIModelInfo]:
    """Return only models with detection or embedding capability."""
    _embedding_caps = {"detection", "embedding"}
    return [m for m in models if set(m.capabilities or []) & _embedding_caps]


@task_handler(id="skier.face_scan.image.task")
async def face_scan_images_task(ctx: ContextInput, params: dict) -> dict:
    """Scan images for faces only — no tag classification."""
    _log.info("Starting face-scan-only task for context: %s", ctx)
    raw_image_ids = get_selected_items(ctx)
    service: RemoteServiceBase = params["service"]

    image_ids: list[int] = []
    for raw in raw_image_ids:
        try:
            image_ids.append(int(raw))
        except (TypeError, ValueError):
            _log.warning("Skipping invalid image id: %s", raw)

    if not image_ids:
        return {
            "status": "noop",
            "message": "No images to process.",
            "processed_ids": [],
            "failed_ids": [],
            "skipped_ids": [],
        }

    try:
        image_metadata = await _with_stash_timeout(
            stash_api.get_image_paths_and_tags_async(image_ids),
            "get_image_paths_and_tags",
        )
    except Exception as exc:
        _log.exception("Failed to fetch image metadata for ids=%s", image_ids)
        detail = _short_error(str(exc) or exc.__class__.__name__)
        return {
            "status": "failed",
            "message": f"Face scan failed while fetching paths ({detail}).",
            "processed_ids": image_ids,
            "failed_ids": image_ids,
            "skipped_ids": [],
        }

    failure_reasons: dict[int, str] = {}
    failed_images: set[int] = set()
    skipped_images: set[int] = set()

    _agg_faces_new = 0
    _agg_faces_matched = 0
    _agg_faces_total = 0
    _agg_new_cluster_ids: list[int] = []
    _agg_matched_cluster_ids: list[int] = []

    valid_paths: dict[int, str] = {}
    reprocess_request_ids: set[int] = set()
    for image_id in image_ids:
        record = (image_metadata or {}).get(image_id) or {}
        path = record.get("path") if isinstance(record, dict) else None
        tags = record.get("tag_ids") if isinstance(record, dict) else None
        if has_ai_reprocess(tags):
            reprocess_request_ids.add(image_id)
        if not path:
            failure_reasons[image_id] = "file path unavailable"
            failed_images.add(image_id)
            continue
        valid_paths[image_id] = path

    if not valid_paths:
        return {
            "status": "failed",
            "message": "No valid images to process.",
            "processed_ids": image_ids,
            "failed_ids": image_ids,
            "skipped_ids": [],
        }

    # Model history check — only compare embedding/detection models
    await update_model_cache(service)
    embedding_models = _filter_embedding_models(list(current_server_models_cache))
    face_scan_fi = service.face_scan_frame_interval

    remote_targets: dict[int, str] = {}
    for image_id, path in valid_paths.items():
        try:
            historical_models = await get_image_model_history_async(
                service=service.name, image_id=image_id,
            )
        except Exception:
            _log.exception("Failed to load image model history for image_id=%s", image_id)
            historical_models = ()

        _, should_reprocess = determine_model_plan(
            current_models=embedding_models,
            previous_models=historical_models,
            current_frame_interval=face_scan_fi,
            current_threshold=FACE_SCAN_THRESHOLD,
        )

        if image_id in reprocess_request_ids:
            _log.info("AI_Reprocess tag present for image_id=%s; forcing face-scan reprocess", image_id)
            should_reprocess = True

        if should_reprocess:
            remote_targets[image_id] = mutate_path_for_plugin(path, service.plugin_name)
        else:
            skipped_images.add(image_id)

    if remote_targets:
        remote_image_ids = list(remote_targets.keys())
        remote_paths = [remote_targets[iid] for iid in remote_image_ids]
        try:
            response = await call_face_scan_images_api(service, remote_paths)
        except Exception:
            _log.exception("Remote face scan failed for %d images", len(remote_image_ids))
            for iid in remote_image_ids:
                failure_reasons[iid] = "remote service request failed"
            failed_images.update(remote_image_ids)
            response = None

        if response is not None:
            result_payload = response.result if isinstance(response.result, list) else []
            models_used = response.models if getattr(response, "models", None) else []
            classifier = build_category_classifier(models_used)
            for idx, image_id in enumerate(remote_image_ids):
                payload = result_payload[idx] if idx < len(result_payload) else {}
                parsed_image = parse_image_result(
                    payload if isinstance(payload, dict) else {}, classifier
                )
                if parsed_image.error:
                    failure_reasons[image_id] = _short_error(parsed_image.error)
                    failed_images.add(image_id)
                    continue

                # Store a lightweight run record for the face scan
                run_id: int | None = None
                try:
                    models_payload = [
                        m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
                        for m in models_used
                    ] if models_used else []
                    run_id = await store_image_run_async(
                        service=service.name,
                        plugin_name=service.plugin_name,
                        image_id=image_id,
                        tag_records=[],
                        input_params={"face_scan_only": True, "threshold": FACE_SCAN_THRESHOLD},
                        requested_models=models_payload,
                    )
                except Exception:
                    _log.exception("Failed to persist face scan run for image_id=%s", image_id)

                if run_id is not None and (parsed_image.detections or parsed_image.regions) and has_embedding_capability(models_used):
                    try:
                        stale = await cleanup_stale_detections_async(
                            entity_type="image",
                            entity_id=image_id,
                            service=service.name,
                            exclude_run_id=run_id,
                        )
                        if stale:
                            _log.debug("Image %s: purged %d stale detection track(s)", image_id, stale)
                    except Exception:
                        _log.exception("Failed to cleanup stale detections for image %s", image_id)
                    try:
                        face_summary = await process_image_detections(
                            run_id=run_id,
                            image_id=image_id,
                            parsed=parsed_image,
                            classifier=classifier,
                            auto_apply_performers=service.auto_apply_performers,
                            auto_threshold=service.face_match_auto_threshold,
                            review_threshold=service.face_match_review_threshold,
                            max_exemplars=service.face_max_exemplars_per_cluster,
                            dedup_threshold=service.face_embedding_dedup_threshold,
                            min_embedding_norm=service.face_min_embedding_norm,
                            min_detection_score=service.face_min_detection_score,
                            hard_min_embedding_norm=service.face_hard_min_embedding_norm,
                            hard_min_detection_score=service.face_hard_min_detection_score,
                            max_embeddings_per_cluster=service.face_max_embeddings_per_cluster,
                        )
                        if face_summary.get("tracks_created"):
                            _agg_faces_new += face_summary["clusters_created"]
                            _agg_faces_matched += face_summary["clusters_matched"]
                            _agg_faces_total += face_summary["tracks_created"]
                            for cid in face_summary.get("new_cluster_ids", []):
                                if cid not in _agg_new_cluster_ids:
                                    _agg_new_cluster_ids.append(cid)
                            for cid in face_summary.get("matched_cluster_ids", []):
                                if cid not in _agg_matched_cluster_ids:
                                    _agg_matched_cluster_ids.append(cid)
                    except Exception:
                        _log.exception("Failed to process face detections for image %s", image_id)

    # Remove AI_Reprocess tag from successfully processed images
    reprocess_cleared = [
        iid for iid in image_ids
        if iid in reprocess_request_ids and iid not in failed_images
    ]
    if reprocess_cleared:
        try:
            await _with_stash_timeout(
                remove_reprocess_tag_from_images(reprocess_cleared),
                "remove_reprocess_tag_from_images",
            )
        except Exception:
            _log.exception("Failed to remove AI_Reprocess tag from %d images", len(reprocess_cleared))

    processed_ids = list(dict.fromkeys(image_ids))
    failed_ids = sorted(failed_images)
    skipped_ids = sorted(skipped_images)
    success_count = len(processed_ids) - len(failed_ids) - len(skipped_ids)
    status = "success"
    if failed_ids:
        status = "failed" if success_count == 0 else "partial"

    if len(processed_ids) == 1:
        image_id = processed_ids[0]
        if image_id in failed_images:
            reason = failure_reasons.get(image_id, "unknown error")
            message = f"Image #{image_id}: face scan failed ({reason})."
        elif image_id in skipped_images:
            message = f"Image #{image_id}: face data already up-to-date, skipped."
        else:
            message = f"Image #{image_id}: scanned {_agg_faces_total} face(s), {_agg_faces_new} new cluster(s)."
    else:
        message = _format_multi_summary("image face scans", success_count, len(failed_ids))

    return {
        "action_type": "face_scan",
        "status": status,
        "message": message,
        "processed_ids": processed_ids,
        "failed_ids": failed_ids,
        "skipped_ids": skipped_ids,
        "failure_reasons": {iid: failure_reasons[iid] for iid in failed_ids},
        "face_summary": {
            "faces_new": _agg_faces_new,
            "faces_matched": _agg_faces_matched,
            "faces_total": _agg_faces_total,
            "new_cluster_ids": _agg_new_cluster_ids,
            "matched_cluster_ids": _agg_matched_cluster_ids,
        },
    }


@task_handler(id="skier.face_scan.scene.task")
async def face_scan_scene_task(ctx: ContextInput, params: dict, task_record: TaskRecord) -> dict:
    """Scan a single scene for faces only — no tag classification."""
    scene_id_raw = ctx.entity_id
    if scene_id_raw is None:
        raise ValueError("Context missing scene entity_id. ctx: %s" % ctx)
    try:
        scene_id = int(scene_id_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid scene_id: {scene_id_raw}") from exc

    service = params["service"]
    try:
        scene_path, scene_tags, scene_duration = await _with_stash_timeout(
            stash_api.get_scene_path_and_tags_and_duration_async(scene_id),
            "get_scene_path_and_tags_and_duration",
        )
    except Exception as exc:
        _log.exception("Failed to load scene metadata for scene_id=%s", scene_id)
        detail = _short_error(str(exc) or exc.__class__.__name__)
        return {
            "scene_id": scene_id,
            "status": "failed",
            "message": f"Scene #{scene_id}: face scan failed ({detail}).",
            "processed_ids": [scene_id],
            "failed_ids": [scene_id],
        }

    remote_scene_path = mutate_path_for_plugin(scene_path or "", service.plugin_name)
    frame_interval = service.face_scan_frame_interval
    vr_scene = is_vr_scene(scene_tags)
    force_reprocess = has_ai_reprocess(scene_tags)

    # --- Model history / reprocessing check ---
    try:
        historical_models = await get_scene_model_history_async(
            service=service.name, scene_id=scene_id,
        )
    except Exception:
        _log.exception("Failed to load scene model history for scene_id=%s", scene_id)
        historical_models = ()

    await update_model_cache(service)
    embedding_models = _filter_embedding_models(list(current_server_models_cache))

    _, should_reprocess = determine_model_plan(
        current_models=embedding_models,
        previous_models=historical_models,
        current_frame_interval=frame_interval,
        current_threshold=FACE_SCAN_THRESHOLD,
    )

    if force_reprocess:
        _log.info("AI_Reprocess tag present for scene_id=%s; forcing face-scan reprocess", scene_id)
        should_reprocess = True

    if not should_reprocess:
        _log.debug("Skipping remote face scan for scene_id=%s; existing face data considered current", scene_id)
        return {
            "action_type": "face_scan",
            "scene_id": scene_id,
            "status": "success",
            "message": f"Scene #{scene_id}: face data already up-to-date, skipped.",
            "processed_ids": [scene_id],
            "failed_ids": [],
            "skipped": True,
            "face_summary": None,
        }

    try:
        response = await call_face_scan_video_api(
            service,
            remote_scene_path,
            frame_interval,
            vr_scene,
        )
    except Exception as exc:
        _log.exception("Face scan API call failed for scene_id=%s", scene_id)
        detail = _short_error(str(exc) or exc.__class__.__name__)
        return {
            "scene_id": scene_id,
            "status": "failed",
            "message": f"Scene #{scene_id}: face scan failed ({detail}).",
            "processed_ids": [scene_id],
            "failed_ids": [scene_id],
        }

    if response is None or response.result is None:
        return {
            "scene_id": scene_id,
            "status": "failed",
            "message": f"Scene #{scene_id}: face scan returned no data.",
            "processed_ids": [scene_id],
            "failed_ids": [scene_id],
        }

    result = response.result
    parsed_frames = None
    face_result_data: dict | None = None

    if result.frames:
        frame_classifier = build_category_classifier(result.models)
        parsed_frames = parse_video_frames(result.frames, frame_classifier)

    # Store a run record so model history tracks this face scan
    run_id: int | None = None
    try:
        result_models_payload = [model.model_dump(exclude_none=True) for model in result.models]
        run_id = await store_scene_run_async(
            service=service.name,
            plugin_name=service.plugin_name,
            scene_id=scene_id,
            input_params={
                "frame_interval": frame_interval,
                "vr_video": vr_scene,
                "face_scan_only": True,
                "threshold": FACE_SCAN_THRESHOLD,
            },
            result_payload=result.model_dump(exclude_none=True),
            requested_models=result_models_payload,
            resolve_reference=lambda _label, _cat: None,
        )
    except Exception:
        _log.exception("Failed to persist face scan run for scene_id=%s", scene_id)

    if parsed_frames and run_id is not None and has_embedding_capability(result.models):
        try:
            stale = await cleanup_stale_detections_async(
                entity_type="scene",
                entity_id=scene_id,
                service=service.name,
                exclude_run_id=run_id,
            )
            if stale:
                _log.debug("Scene %s: purged %d stale detection track(s)", scene_id, stale)
        except Exception:
            _log.exception("Failed to cleanup stale detections for scene %s", scene_id)

        try:
            frame_classifier = build_category_classifier(result.models)
            face_summary = await process_video_detections(
                run_id=run_id,
                scene_id=scene_id,
                parsed_frames=parsed_frames,
                frame_interval=result.frame_interval,
                classifier=frame_classifier,
                auto_apply_performers=service.auto_apply_performers,
                auto_threshold=service.face_match_auto_threshold,
                review_threshold=service.face_match_review_threshold,
                max_exemplars=service.face_max_exemplars_per_cluster,
                max_embeddings_per_track=service.face_max_embeddings_per_track,
                dedup_threshold=service.face_embedding_dedup_threshold,
                min_embedding_norm=service.face_min_embedding_norm,
                min_detection_score=service.face_min_detection_score,
                hard_min_embedding_norm=service.face_hard_min_embedding_norm,
                hard_min_detection_score=service.face_hard_min_detection_score,
                max_embeddings_per_cluster=service.face_max_embeddings_per_cluster,
            )
            if face_summary.get("tracks_created"):
                auto_merged = face_summary.get("clusters_auto_merged", 0)
                face_result_data = {
                    "faces_new": len(face_summary.get("new_cluster_ids", [])),
                    "faces_matched": face_summary["clusters_matched"],
                    "faces_total": face_summary["tracks_created"],
                    "faces_auto_merged": auto_merged,
                    "new_cluster_ids": face_summary.get("new_cluster_ids", []),
                    "matched_cluster_ids": face_summary.get("matched_cluster_ids", []),
                }
                _log.debug(
                    "Scene %s face scan: %d detected, %d matched, %d new, %d auto-merged",
                    scene_id, face_summary["tracks_created"],
                    face_summary["clusters_matched"],
                    len(face_summary.get("new_cluster_ids", [])),
                    auto_merged,
                )
        except Exception:
            _log.exception("Failed to process face detections for scene %s", scene_id)

    # Remove AI_Reprocess tag after successful processing
    if force_reprocess:
        try:
            await remove_reprocess_tag_from_scene(scene_id)
        except Exception:
            _log.exception("Failed to remove AI_Reprocess tag from scene %s", scene_id)

    if face_result_data:
        message = (
            f"Scene #{scene_id}: {face_result_data['faces_total']} face(s) detected, "
            f"{face_result_data['faces_matched']} matched, "
            f"{face_result_data['faces_new']} new cluster(s)."
        )
    else:
        message = f"Scene #{scene_id}: face scan complete (no faces detected)."

    return {
        "action_type": "face_scan",
        "scene_id": scene_id,
        "status": "success",
        "message": message,
        "processed_ids": [scene_id],
        "failed_ids": [],
        "face_summary": face_result_data,
    }


async def face_scan_scenes(service: RemoteServiceBase, ctx: ContextInput, params: dict, task_record: TaskRecord):
    """Controller for multi-scene face scanning. Spawns chunked child tasks."""
    caps = getattr(service, '_backend_capabilities', None)
    if caps is not None and not caps.get('face_recognition'):
        raise RuntimeError(
            "Face recognition is not supported by the connected AI backend. "
            "Enable a face recognition model in your backend configuration."
        )
    selected_items = get_selected_items(ctx)
    params["service"] = service
    if not selected_items:
        return {
            "status": "noop",
            "message": "No scenes to process.",
            "scenes_requested": 0,
            "scenes_completed": 0,
            "scenes_failed": 0,
        }
    if len(selected_items) == 1:
        if not ctx.entity_id:
            ctx.entity_id = str(selected_items[0])
        return await face_scan_scene_task(ctx, params, task_record)

    task_priority = TaskPriority.low
    if ctx.is_detail_view:
        task_priority = TaskPriority.high
    elif ctx.selected_ids and len(ctx.selected_ids) >= 1:
        task_priority = TaskPriority.normal
    elif ctx.visible_ids and len(ctx.visible_ids) >= 1:
        task_priority = TaskPriority.normal

    spawn_result = await spawn_chunked_tasks(
        parent_task=task_record,
        parent_context=ctx,
        handler=face_scan_scene_task,
        items=selected_items,
        chunk_size=1,
        params=params,
        priority=task_priority,
        hold_children=True,
    )
    child_ids = spawn_result.get("spawned", [])
    success = 0
    failed = 0
    for child_id in child_ids:
        child = task_manager.get(child_id)
        if child is None:
            continue
        if child.status == TaskStatus.failed:
            failed += 1
            continue
        child_result = getattr(child, "result", None)
        if isinstance(child_result, dict) and child_result.get("status") == "failed":
            failed += 1
        else:
            success += 1

    total_requested = len(selected_items)
    accounted = success + failed
    if accounted < len(child_ids):
        failed += len(child_ids) - accounted
    if accounted < total_requested:
        failed += total_requested - accounted

    failed = min(failed, total_requested)
    success = max(total_requested - failed, 0)

    status = "success"
    if failed:
        status = "failed" if success == 0 else "partial"
    message = _format_multi_summary("scene face scans", success, failed)

    return {
        "action_type": "face_scan",
        "status": status,
        "message": message,
        "scenes_requested": total_requested,
        "scenes_completed": success,
        "scenes_failed": failed,
        "spawned": child_ids,
        "count": len(child_ids),
        "held": spawn_result.get("held", False),
    }


async def face_scan_images(service: RemoteServiceBase, ctx: ContextInput, params: dict, task_record: TaskRecord):
    """Controller for multi-image face scanning. Spawns chunked child tasks."""
    caps = getattr(service, '_backend_capabilities', None)
    if caps is not None and not caps.get('face_recognition'):
        raise RuntimeError(
            "Face recognition is not supported by the connected AI backend. "
            "Enable a face recognition model in your backend configuration."
        )
    selected_items = get_selected_items(ctx)
    params["service"] = service
    if not selected_items:
        return {
            "status": "noop",
            "message": "No images to process.",
            "images_requested": 0,
            "images_completed": 0,
            "images_failed": 0,
        }
    if len(selected_items) <= MAX_FACE_SCAN_IMAGES_PER_REQUEST:
        return await face_scan_images_task(ctx, params)

    task_priority = TaskPriority.low
    if ctx.is_detail_view:
        task_priority = TaskPriority.high
    elif ctx.selected_ids and len(ctx.selected_ids) >= 1:
        task_priority = TaskPriority.normal
    elif ctx.visible_ids and len(ctx.visible_ids) >= 1:
        task_priority = TaskPriority.normal

    spawn_result = await spawn_chunked_tasks(
        parent_task=task_record,
        parent_context=ctx,
        handler=face_scan_images_task,
        items=selected_items,
        chunk_size=MAX_FACE_SCAN_IMAGES_PER_REQUEST,
        params=params,
        priority=task_priority,
        hold_children=True,
    )
    child_ids = spawn_result.get("spawned", [])
    total_requested = len(selected_items)
    success = 0
    failed = 0
    for child_id in child_ids:
        child = task_manager.get(child_id)
        if child is None:
            continue
        chunk_total = 0
        try:
            if child.context.selected_ids:
                chunk_total = len(child.context.selected_ids)
            elif child.context.entity_id:
                chunk_total = 1
        except Exception:
            chunk_total = 0
        if child.status == TaskStatus.failed:
            failed += chunk_total or 1
            continue
        child_result = getattr(child, "result", None)
        if isinstance(child_result, dict):
            processed = child_result.get("processed_ids") or []
            failed_ids = child_result.get("failed_ids") or []
            success += max(len(processed) - len(failed_ids), 0)
            failed += len(failed_ids)
        else:
            failed += chunk_total or 1

    failed = min(failed, total_requested)
    success = max(total_requested - failed, 0)

    status = "success"
    if failed:
        status = "failed" if success == 0 else "partial"
    message = _format_multi_summary("image face scans", success, failed)

    return {
        "action_type": "face_scan",
        "status": status,
        "message": message,
        "images_requested": total_requested,
        "images_completed": success,
        "images_failed": failed,
        "spawned": child_ids,
        "count": len(child_ids),
        "held": spawn_result.get("held", False),
    }


# ------------------------------------------------------------------
# Tag configuration methods (for plugin endpoints)
# ------------------------------------------------------------------


async def get_available_tags_data(service: RemoteServiceBase) -> dict:
    """Get available tags from CSV file with full settings.

    Returns:
        dict with 'tags' (full settings), 'models', and 'defaults' keys.
    """
    
    _log = logging.getLogger(__name__)
    
    # Get tag config
    tag_config_obj = get_tag_configuration()
    
    # Read tags directly from CSV file
    tags_list = []
    defaults = {}
    csv_path = tag_config_obj.source_path
    
    if not csv_path.exists():
        _log.warning("Tag settings CSV file does not exist at %s", csv_path)
        return {'tags': [], 'models': [], 'defaults': {}}
    
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                _log.warning("Tag settings CSV file is missing a header row")
                return {'tags': [], 'models': [], 'defaults': {}}
            
            for row in reader:
                # Get tag name from CSV
                tag_name = (row.get('tag_name') or row.get('tag') or '').strip()
                
                # Extract default values from __default__ row
                if tag_name.lower() == '__default__':
                    defaults['required_scene_tag_duration'] = row.get('RequiredSceneTagDuration', '').strip()
                    defaults['min_marker_duration'] = row.get('min_marker_duration', '').strip()
                    defaults['max_gap'] = row.get('max_gap', '').strip()
                    defaults['markers_enabled'] = row.get('markers_enabled', 'TRUE').strip().upper() == 'TRUE'
                    continue
                
                # Skip empty rows
                if not tag_name or tag_name.lower() in {'', '*', 'default', 'unused1', 'unused2', 'unused3', 'unused4'}:
                    continue
                
                # Get resolved settings for this tag
                settings = tag_config_obj.resolve(tag_name)
                
                # Get categories from CSV row (pipe-delimited list)
                raw_category = row.get('category', '').strip()
                if raw_category:
                    categories = [c.strip() for c in raw_category.split('|') if c.strip()]
                else:
                    categories = ['Other']
                
                # Format required_scene_tag_duration
                req_duration_str = None
                if settings.required_scene_tag_duration:
                    if settings.required_scene_tag_duration.unit == 'percent':
                        req_duration_str = f"{settings.required_scene_tag_duration.value}%"
                    else:
                        req_duration_str = str(settings.required_scene_tag_duration.value)
                
                # Add tag with full settings
                tags_list.append({
                    'tag': tag_name,
                    'name': tag_name,  # For compatibility
                    'categories': categories,
                    'scene_tag_enabled': settings.scene_tag_enabled,
                    'markers_enabled': settings.markers_enabled,
                    'image_enabled': settings.image_enabled,
                    'required_scene_tag_duration': req_duration_str,
                    'min_marker_duration': settings.min_marker_duration,
                    'max_gap': settings.max_gap,
                })
    except Exception as exc:
        _log.exception("Failed to read tags from CSV file %s: %s", csv_path, exc)
        return {'tags': [], 'models': [], 'defaults': {}, 'error': f'Failed to read CSV: {str(exc)}'}
    
    # Fetch active models from nsfw backend
    active_models = []
    loaded_categories = set()
    try:
        active_models_list = await get_active_scene_models(service)
        if active_models_list:
            for model in active_models_list:
                # Convert AIModelInfo to dict for JSON serialization
                model_dict = {
                    'name': model.name,
                    'identifier': model.identifier,
                    'version': model.version,
                    'categories': model.categories,
                    'type': model.type,
                }
                active_models.append(model_dict)
                # Extract all categories from this model
                if model.categories:
                    loaded_categories.update(model.categories)
    except Exception as exc:
        # If backend is unavailable, log warning but continue (graceful degradation)
        _log.warning("Failed to fetch active models from nsfw backend: %s. Showing all tags.", exc)
    
    return {
        'tags': tags_list,
        'models': active_models,
        'loaded_categories': list(loaded_categories),
        'defaults': defaults
    }


def update_tag_settings(tag_settings: dict) -> dict:
    """Update full tag settings for multiple tags.
    
    Args:
        tag_settings: Dictionary mapping tag names (normalized, lowercase) to dicts with:
            - scene_tag_enabled: bool (optional)
            - markers_enabled: bool (optional)
            - image_enabled: bool (optional)
            - required_scene_tag_duration: str (optional, e.g., "15", "15s", "35%")
            - min_marker_duration: float (optional)
            - max_gap: float (optional)
    
    Returns:
        dict with 'status' and 'updated' count
    """
    tag_config_obj = get_tag_configuration()
    
    # Convert to format expected by tag_config
    settings_map = {}
    for tag_name, settings in tag_settings.items():
        settings_map[tag_name] = {
            'scene_tag_enabled': settings.get('scene_tag_enabled'),
            'markers_enabled': settings.get('markers_enabled'),
            'image_enabled': settings.get('image_enabled'),
            'required_scene_tag_duration': settings.get('required_scene_tag_duration'),
            'min_marker_duration': settings.get('min_marker_duration'),
            'max_gap': settings.get('max_gap'),
        }
    
    tag_config_obj.update_tag_settings(settings_map)
    return {'status': 'ok', 'updated': len(tag_settings)}