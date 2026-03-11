from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import quote

from stash_ai_server.services.registry import ServiceBase, services
from stash_ai_server.actions.registry import action
from stash_ai_server.actions.models import ContextRule, ContextInput
from stash_ai_server.tasks.models import TaskRecord, TaskPriority, TaskStatus, TaskSpec
from stash_ai_server.tasks.helpers import spawn_chunked_tasks, task_handler
from stash_ai_server.tasks.manager import manager as task_manager
from stash_ai_server.utils.stash_api import stash_api
from stash_ai_server.utils.path_mutation import mutate_path_for_plugin

from . import stash_helpers
from . import CODEC_FAMILIES

_log = logging.getLogger(__name__)

# Default worker URL — the reencode_worker sidecar on the same host network
_DEFAULT_WORKER_URL = "http://localhost:4154"


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_DEFAULT_SKIP_CODECS = ["hevc", "av1", "vp9"]


def _coerce_skip_codecs(adv: dict) -> list[str]:
    """Read skip_codecs from advanced blob with backwards compat from skip_hevc."""
    raw = adv.get("skip_codecs")
    if isinstance(raw, list):
        return [c for c in raw if isinstance(c, str)]
    # Backwards compat: fall back to old skip_hevc boolean
    if "skip_hevc" in adv:
        if _coerce_bool(adv["skip_hevc"], True):
            return ["hevc", "av1", "vp9"]
        return []
    return list(_DEFAULT_SKIP_CODECS)


def _get_selected_items(ctx: ContextInput) -> list[str]:
    """Extract selected item IDs from context."""
    if ctx.entity_id:
        return [ctx.entity_id]
    if ctx.selected_ids:
        return list(ctx.selected_ids)
    if ctx.visible_ids:
        return list(ctx.visible_ids)
    return []


# ---------------------------------------------------------------------------
# Worker HTTP client helpers (stdlib only — no aiohttp in backend container)
# ---------------------------------------------------------------------------
async def _worker_request(worker_url: str, method: str, path: str, json_body: dict | None = None) -> dict:
    """Make an HTTP request to the reencode worker sidecar using stdlib."""
    import json as _json
    import urllib.request

    url = f"{worker_url.rstrip('/')}{path}"

    def _do_request():
        if json_body is not None:
            data = _json.dumps(json_body).encode("utf-8")
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header("Content-Type", "application/json")
        else:
            req = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return _json.loads(resp.read())

    return await asyncio.to_thread(_do_request)


async def _worker_health(worker_url: str) -> dict | None:
    """Check worker health. Returns response dict or None if unreachable."""
    try:
        return await _worker_request(worker_url, "GET", "/health")
    except Exception:
        return None


async def _submit_encode_job(worker_url: str, file_path: str, settings: dict, stash_path: str = "") -> str | None:
    """Submit an encode job to the worker. Returns job_id or None on error."""
    try:
        resp = await _worker_request(worker_url, "POST", "/encode", {
            "file_path": file_path,
            "settings": settings,
            "stash_path": stash_path,
        })
        return resp.get("job_id")
    except Exception as exc:
        _log.error("Failed to submit encode job: %s", exc)
        return None


async def _recover_worker_jobs(service) -> None:
    """Reconnect to in-flight and uncollected worker jobs after a backend restart.

    Creates a real controller task so the dashboard shows live progress.
    For already-completed jobs, triggers Stash rescans immediately.
    """
    worker_url = service.worker_url
    try:
        resp = await _worker_request(worker_url, "GET", "/jobs")
    except Exception as exc:
        _log.debug("Recovery: worker unreachable: %s", exc)
        return

    jobs = resp.get("jobs", [])
    running_jobs = [j for j in jobs if not j.get("done") and not j.get("collected")]
    completed_jobs = [j for j in jobs if j.get("done") and not j.get("collected")]

    if not running_jobs and not completed_jobs:
        return

    _log.info("Recovery: found %d running, %d completed uncollected job(s)",
              len(running_jobs), len(completed_jobs))

    # Handle completed-uncollected jobs: trigger rescans
    for job in completed_jobs:
        result = job.get("result", {})
        stash_path = job.get("stash_path", "")
        job_id = job.get("job_id", "?")
        if stash_path and result.get("success"):
            rescan_path = result.get("output_path") or stash_path
            _log.info("Recovery: rescan for completed job %s → %s", job_id, rescan_path)
            try:
                await stash_helpers.trigger_rescan(rescan_path)
            except Exception as exc:
                _log.warning("Recovery: rescan failed for %s: %s", rescan_path, exc)
        # Mark collected
        try:
            await _worker_request(worker_url, "GET", f"/status/{job_id}")
        except Exception:
            pass

    # Handle running jobs: create a controller task that polls them
    if not running_jobs:
        return

    # Submit a recovery controller task
    spec = TaskSpec(id="jiwenji.reencode.recovery", service="jiwenji_reencode")
    ctx = ContextInput(page="scenes", entity_id=None, is_detail_view=False, selected_ids=[])

    async def _recovery_controller(ctx, params, task_record):
        return await _poll_recovered_jobs(worker_url, running_jobs, task_record, service)

    task = task_manager.submit(spec, _recovery_controller, ctx, {}, TaskPriority.high)
    _log.info("Recovery: created controller task %s for %d running job(s)", task.id, len(running_jobs))


async def _poll_recovered_jobs(worker_url: str, jobs: list, task_record: TaskRecord, service) -> dict:
    """Poll recovered running jobs until all finish, emitting dashboard-compatible progress."""
    # Restore batch state from worker (total scenes, prior completed/failed/skipped/savings)
    batch_state = {}
    try:
        batch_state = await _worker_request(worker_url, "GET", "/batch")
    except Exception:
        pass

    job_ids = [j["job_id"] for j in jobs]
    stash_paths = {j["job_id"]: j.get("stash_path", "") for j in jobs}
    done_set: set[str] = set()

    # Use batch state as baseline — these include counts from before the crash
    batch_total = batch_state.get("total", len(jobs))
    base_success = batch_state.get("success", 0)
    base_failed = batch_state.get("failed", 0)
    base_skipped = batch_state.get("skipped", 0)
    base_completed = batch_state.get("completed", 0)
    base_savings_mb = batch_state.get("savings_mb", 0.0)

    # Track new completions from recovered jobs only
    new_success = 0
    new_failed = 0
    new_skipped = 0
    new_savings_bytes = 0

    num_recovered_jobs = len(job_ids)

    while len(done_set) < num_recovered_jobs:
        if getattr(task_record, "cancel_requested", False):
            for jid in job_ids:
                if jid not in done_set:
                    try:
                        await _worker_request(worker_url, "POST", f"/cancel/{jid}")
                    except Exception:
                        pass
            break

        workers_detail = []
        for jid in job_ids:
            if jid in done_set:
                continue
            try:
                resp = await _worker_request(worker_url, "GET", f"/status/{jid}")
            except Exception:
                continue

            if resp.get("done"):
                done_set.add(jid)
                result = resp.get("result", {})
                stash_path = stash_paths.get(jid, "")
                if result.get("success"):
                    new_success += 1
                    orig = result.get("original_size", 0)
                    new = result.get("new_size", 0)
                    if orig and new:
                        new_savings_bytes += orig - new
                    if stash_path:
                        rescan_path = result.get("output_path") or stash_path
                        try:
                            await stash_helpers.trigger_rescan(rescan_path)
                        except Exception as exc:
                            _log.warning("Recovery: rescan failed for %s: %s", rescan_path, exc)
                elif result.get("skipped"):
                    new_skipped += 1
                else:
                    new_failed += 1
            else:
                workers_detail.append({
                    "percent": resp.get("progress", 0.0),
                    "fps": resp.get("fps", 0.0),
                    "speed": resp.get("speed", ""),
                    "filename": resp.get("filename", ""),
                })

        # Combine baseline (pre-crash) + new completions from recovered jobs
        total_completed = base_completed + len(done_set)
        total_success = base_success + new_success
        total_failed = base_failed + new_failed
        total_skipped = base_skipped + new_skipped
        total_savings_mb = base_savings_mb + round(new_savings_bytes / (1024 * 1024), 1)

        task_manager.emit_progress(task_record, {
            "completed": total_completed,
            "total": batch_total,
            "running": num_recovered_jobs - len(done_set),
            "failed": total_failed,
            "skipped": total_skipped,
            "success": total_success,
            "status_line": f"{total_completed}/{batch_total} (recovered)",
            "savings_mb": total_savings_mb,
            "workers": workers_detail,
        })

        if len(done_set) < num_recovered_jobs:
            await asyncio.sleep(2.0)

    final_success = base_success + new_success
    final_failed = base_failed + new_failed
    final_skipped = base_skipped + new_skipped
    final_savings = base_savings_mb + round(new_savings_bytes / (1024 * 1024), 1)
    return {
        "status": "recovered",
        "message": f"Recovery complete: {final_success} succeeded, {final_failed} failed, {final_skipped} skipped. Savings: {final_savings} MB",
        "scenes_completed": final_success,
        "scenes_failed": final_failed,
        "scenes_skipped": final_skipped,
        "total_savings_mb": final_savings,
    }


async def _poll_encode_job(
    worker_url: str,
    job_id: str,
    task_record: TaskRecord,
    cancel_check,
    poll_interval: float = 2.0,
) -> dict:
    """Poll worker for job status until done or cancelled."""
    MAX_POLL_FAILURES = 30
    poll_failures = 0

    while True:
        if cancel_check():
            # Cancel on worker side too
            try:
                await _worker_request(worker_url, "POST", f"/cancel/{job_id}")
            except Exception:
                pass
            return {"success": False, "error": "Cancelled"}

        try:
            resp = await _worker_request(worker_url, "GET", f"/status/{job_id}")
            poll_failures = 0  # reset on success
        except Exception as exc:
            poll_failures += 1
            if poll_failures >= MAX_POLL_FAILURES:
                _log.error(
                    "Worker unreachable for %d consecutive polls, aborting job %s",
                    poll_failures, job_id,
                )
                return {"success": False, "error": "Worker unreachable"}
            _log.warning(
                "Failed to poll job %s (%d/%d): %s",
                job_id, poll_failures, MAX_POLL_FAILURES, exc,
            )
            await asyncio.sleep(poll_interval)
            continue

        # Worker restarted and lost this job — it returns {"error": "Unknown job_id"}
        if resp.get("error") and not resp.get("done"):
            _log.error("Worker lost job %s: %s", job_id, resp["error"])
            return {"success": False, "error": f"Worker lost job: {resp['error']}"}

        progress = resp.get("progress", 0)
        pct = round(progress, 1)
        task_manager.emit_progress(task_record, {
            "percent": pct,
            "fps": resp.get("fps", 0.0),
            "speed": resp.get("speed", ""),
            "filename": resp.get("filename", ""),
            "status_line": f"{pct:.1f}%",
        })

        if resp.get("done"):
            return resp.get("result", {"success": False, "error": "No result returned"})

        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Task handler — processes a single scene
# ---------------------------------------------------------------------------
@task_handler(id="jiwenji.reencode.scene.task", service="jiwenji_reencode")
async def reencode_scene_task(ctx: ContextInput, params: dict, task_record: TaskRecord) -> dict:
    scene_id_raw = ctx.entity_id
    if scene_id_raw is None:
        raise ValueError("Context missing scene entity_id")
    scene_id = int(scene_id_raw)

    service: ReencodeService = params["service"]
    settings = service.get_encode_settings()
    worker_url = service.worker_url

    # 1. Get file info from Stash
    file_info = await stash_helpers.get_scene_file_info(scene_id)
    if not file_info or not file_info.get("path"):
        return {
            "scene_id": scene_id,
            "status": "failed",
            "message": f"Scene #{scene_id}: could not retrieve file info from Stash.",
        }

    stash_path = file_info["path"]
    scene_tags = file_info.get("tags") or set()
    fail_tag = settings.get("reencode_failed_tag", "reencode_failed")

    # 2. Apply path mapping (Stash Windows path → worker container path)
    worker_path = mutate_path_for_plugin(stash_path, service.plugin_name)

    # 3. Skip codec check using Stash-reported codec (avoids extra ffprobe)
    skip_codecs = settings.get("skip_codecs") or []
    codec = (file_info.get("video_codec") or "").lower()
    if codec and skip_codecs:
        for family_key in skip_codecs:
            aliases = CODEC_FAMILIES.get(family_key, frozenset())
            if codec in aliases:
                # Even though we skip re-encoding, still chain AI tagging if enabled
                tag_queued = False
                needs_tagging = _coerce_bool(settings.get("tag_after_reencode"), True)
                if needs_tagging:
                    tag_in_parallel = _coerce_bool(settings.get("tag_in_parallel"), True)
                    parent_id = getattr(task_record, "group_id", None) or task_record.id
                    group = parent_id if tag_in_parallel else None
                    tag_queued = await _chain_ai_tagging(scene_id, stash_path, group_id=group)
                return {
                    "scene_id": scene_id,
                    "status": "skipped",
                    "message": f"Scene #{scene_id}: already {family_key.upper()}, skipped.",
                    "skipped": True,
                    "tag_queued": tag_queued,
                    "needs_tagging": needs_tagging and not tag_queued,
                    "target_scene_id": scene_id,
                }

    # 3b. Skip scenes tagged with the failure tag (unless disabled)
    if _coerce_bool(settings.get("skip_failed_tag"), True) and fail_tag:
        if fail_tag.lower() in scene_tags:
            return {
                "scene_id": scene_id,
                "status": "skipped",
                "message": f"Scene #{scene_id}: tagged '{fail_tag}', skipped.",
                "skipped": True,
            }

    # 4. Submit encode job to worker sidecar
    job_id = await _submit_encode_job(worker_url, worker_path, settings, stash_path=stash_path)
    if not job_id:
        return {
            "scene_id": scene_id,
            "status": "failed",
            "message": (
                f"Scene #{scene_id}: failed to submit encode job to worker at {worker_url}. "
                f"Is the reencode_worker container running?"
            ),
        }

    # 5. Poll for completion
    def cancel_cb() -> bool:
        return getattr(task_record, "cancel_requested", False)

    result = await _poll_encode_job(worker_url, job_id, task_record, cancel_cb)

    if result.get("skipped"):
        # Worker skipped (e.g. low bitrate) — still chain AI tagging if enabled
        tag_queued = False
        needs_tagging = _coerce_bool(settings.get("tag_after_reencode"), True)
        if needs_tagging:
            tag_in_parallel = _coerce_bool(settings.get("tag_in_parallel"), True)
            parent_id = getattr(task_record, "group_id", None) or task_record.id
            group = parent_id if tag_in_parallel else None
            tag_queued = await _chain_ai_tagging(scene_id, stash_path, group_id=group)
        return {
            "scene_id": scene_id,
            "status": "skipped",
            "message": f"Scene #{scene_id}: {result.get('skip_reason', 'skipped')}.",
            "skipped": True,
            "tag_queued": tag_queued,
            "needs_tagging": needs_tagging and not tag_queued,
            "target_scene_id": scene_id,
        }

    if not result.get("success"):
        # Tag the scene as failed if configured
        tag_on_failure = _coerce_bool(settings.get("tag_on_failure"), True)
        fail_tag = settings.get("reencode_failed_tag", "reencode_failed")
        if tag_on_failure and fail_tag:
            try:
                await stash_helpers.tag_scene(scene_id, fail_tag)
            except Exception as exc:
                _log.warning("Failed to tag scene %s with %r: %s", scene_id, fail_tag, exc)
        return {
            "scene_id": scene_id,
            "status": "failed",
            "message": f"Scene #{scene_id}: encode failed — {result.get('error', 'unknown')}.",
        }

    # 6. Determine output Stash path for rescan
    #    If the worker changed the container format (e.g. .wmv → .mp4),
    #    output_path in the result tells us the new filename.
    worker_output_path = result.get("output_path")
    suffix = settings.get("output_suffix", "")
    delete_after = settings.get("delete_after_convert", True)
    if not suffix and not delete_after:
        suffix = "_hevc"

    p = PureWindowsPath(stash_path) if "\\" in stash_path else PurePosixPath(stash_path)

    if worker_output_path:
        # Container format changed — derive Stash path from the worker's output
        worker_out = PurePosixPath(worker_output_path)
        output_stash_path = str(p.with_name(worker_out.name))
    elif suffix:
        output_stash_path = str(p.with_name(p.stem + suffix + p.suffix))
    else:
        output_stash_path = stash_path

    # 7. Remove failure tag if present (scene succeeded now)
    if fail_tag and fail_tag.lower() in scene_tags:
        try:
            await stash_helpers.untag_scene(scene_id, fail_tag)
        except Exception as exc:
            _log.warning("Failed to remove tag %r from scene %s: %s", fail_tag, scene_id, exc)

    # 8. Trigger Stash rescan
    await stash_helpers.trigger_rescan(output_stash_path)

    # 9. Suffix mode: copy metadata to new scene
    tag_queued = False
    target_scene_id = scene_id

    if suffix and output_stash_path != stash_path and _coerce_bool(settings.get("copy_metadata_on_suffix"), True):
        new_scene_id = await _poll_for_new_scene(output_stash_path, timeout=30)
        if new_scene_id:
            await stash_helpers.copy_scene_metadata(scene_id, new_scene_id)
            target_scene_id = new_scene_id
        else:
            _log.warning("New scene not found for %s after rescan; metadata copy skipped", output_stash_path)

    # 9. Tag after re-encode (if enabled)
    #    When tag_in_parallel is False, the controller defers tagging until all
    #    encodes finish, so we only chain here when parallel mode is ON.
    needs_tagging = _coerce_bool(settings.get("tag_after_reencode"), True)
    tag_in_parallel = _coerce_bool(settings.get("tag_in_parallel"), True)
    if needs_tagging and tag_in_parallel:
        # Use parent's group_id so chained tag task is a child of the same controller,
        # bypassing the global concurrency limit (user opted in to parallel).
        parent_id = getattr(task_record, "group_id", None) or task_record.id
        tag_queued = await _chain_ai_tagging(target_scene_id, output_stash_path, group_id=parent_id)

    original_size = result.get("original_size", 0)
    new_size = result.get("new_size", 0)
    savings_pct = result.get("savings_pct", 0)
    savings_mb = round((original_size - (new_size or 0)) / (1024 * 1024), 1)
    return {
        "scene_id": scene_id,
        "status": "success",
        "message": (
            f"Scene #{scene_id}: re-encoded with {result.get('method_used')}, "
            f"saved {savings_pct}% ({savings_mb} MB)."
        ),
        "original_size": original_size,
        "new_size": new_size,
        "savings_pct": savings_pct,
        "method_used": result.get("method_used"),
        "tag_queued": tag_queued,
        "needs_tagging": needs_tagging and not tag_queued,
        "target_scene_id": target_scene_id,
    }


async def _poll_for_new_scene(file_path: str, timeout: float = 30) -> int | None:
    """Poll Stash for a new scene at the given path."""
    elapsed = 0.0
    while elapsed < timeout:
        scene_id = await stash_helpers.find_scene_by_path(file_path)
        if scene_id is not None:
            return scene_id
        await asyncio.sleep(2)
        elapsed += 2
    return None


async def _chain_ai_tagging(scene_id: int, file_path: str, *, group_id: str | None = None) -> bool:
    """Chain into skier_aitagging after re-encode if the plugin is available.

    We already have the scene ID from the reencode result, so we submit the
    tagging job directly — no need to wait for Stash's rescan to finish.

    When group_id is set, the tagging task becomes a child of the reencode
    controller and bypasses the global concurrency limit (the user explicitly
    opted in to parallel execution via the ``tag_in_parallel`` setting).
    """
    try:
        from stash_ai_server.actions.registry import registry as action_registry

        tag_ctx = ContextInput(
            page="scenes",
            entity_id=str(scene_id),
            is_detail_view=True,
        )
        resolved = action_registry.resolve("skier.ai_tag.scene", tag_ctx)
        if resolved:
            definition, handler = resolved
            task_manager.submit(definition, handler, tag_ctx, {}, TaskPriority.normal, group_id=group_id)
            _log.info("Queued AI tagging for scene %s (group=%s)", scene_id, group_id)
            return True
        else:
            _log.warning("skier.ai_tag.scene action not found; AI tagging skipped")
            return False
    except Exception as exc:
        _log.warning("Failed to chain AI tagging for scene %s: %s", scene_id, exc)
        return False


# ---------------------------------------------------------------------------
# Controller — single or batch
# ---------------------------------------------------------------------------
async def reencode_scenes(service: ReencodeService, ctx: ContextInput, params: dict, task_record: TaskRecord):
    selected_items = _get_selected_items(ctx)
    params["service"] = service
    settings = service.get_encode_settings()
    tag_in_parallel = _coerce_bool(settings.get("tag_in_parallel"), True)

    if not selected_items:
        return {
            "status": "noop",
            "message": "No scenes to process.",
            "scenes_requested": 0,
            "scenes_completed": 0,
            "scenes_failed": 0,
            "scenes_skipped": 0,
        }

    # Single scene: run directly
    if len(selected_items) == 1:
        if not ctx.entity_id:
            ctx.entity_id = str(selected_items[0])
        result = await reencode_scene_task(ctx, params, task_record)
        # Deferred tagging for single scene (when parallel is off)
        if isinstance(result, dict) and result.get("needs_tagging") and result.get("target_scene_id"):
            await _chain_ai_tagging(result["target_scene_id"], "")
            result["tag_queued"] = True
            result["needs_tagging"] = False
        return result

    # Multiple scenes: spawn child tasks (don't hold — we'll poll ourselves for richer progress)
    task_priority = TaskPriority.low
    if ctx.is_detail_view:
        task_priority = TaskPriority.high
    elif ctx.selected_ids and len(ctx.selected_ids) >= 1:
        task_priority = TaskPriority.normal

    spawn_result = await spawn_chunked_tasks(
        parent_task=task_record,
        parent_context=ctx,
        handler=reencode_scene_task,
        items=selected_items,
        chunk_size=1,
        params=params,
        priority=task_priority,
        hold_children=False,
        mark_parent_controller=False,
    )
    child_ids = spawn_result.get("spawned", [])
    total = len(child_ids)

    # Custom hold loop with rich progress reporting
    while True:
        children = [task_manager.get(cid) for cid in child_ids]
        children = [c for c in children if c is not None]

        done = 0
        running_workers = []
        workers_detail = []
        failed_count = 0
        skipped_count = 0
        success_count = 0
        total_savings_bytes = 0
        total_original_bytes = 0

        for idx, child in enumerate(children):
            if child.status in (TaskStatus.completed, TaskStatus.failed, TaskStatus.cancelled):
                done += 1
                child_result = getattr(child, "result", None)
                if child.status == TaskStatus.failed:
                    failed_count += 1
                elif isinstance(child_result, dict):
                    st = child_result.get("status", "")
                    if st == "failed":
                        failed_count += 1
                    elif st == "skipped" or child_result.get("skipped"):
                        skipped_count += 1
                    else:
                        success_count += 1
                        orig = child_result.get("original_size", 0)
                        new = child_result.get("new_size", 0)
                        if orig and new:
                            total_original_bytes += orig
                            total_savings_bytes += orig - new
                else:
                    failed_count += 1
            elif child.status == TaskStatus.running:
                wp = _get_child_progress_dict(child)
                running_workers.append(f"w{len(running_workers) + 1}:{wp['percent']:.1f}%")
                workers_detail.append(wp)

        # Build status line: "done: 3/10|w1:45.0%|w2:12.0%|failed:1"
        parts = [f"done: {done}/{total}"]
        if running_workers:
            parts.extend(running_workers)
        if failed_count:
            parts.append(f"failed: {failed_count}")
        if skipped_count:
            parts.append(f"skipped: {skipped_count}")
        status_line = "|".join(parts)

        progress_payload = {
            "completed": done,
            "total": total,
            "running": len(running_workers),
            "failed": failed_count,
            "skipped": skipped_count,
            "success": success_count,
            "status_line": status_line,
            "savings_mb": round(total_savings_bytes / (1024 * 1024), 1),
            "workers": workers_detail,
        }
        task_manager.emit_progress(task_record, progress_payload)

        # Persist batch state on worker for crash recovery
        try:
            await _worker_request(service.worker_url, "PUT", "/batch", progress_payload)
        except Exception:
            pass

        pending = [c for c in children if c.status not in (TaskStatus.completed, TaskStatus.failed, TaskStatus.cancelled)]
        if not pending:
            break
        if getattr(task_record, "cancel_requested", False):
            break
        await asyncio.sleep(1.0)

    # Final aggregation
    total_requested = len(selected_items)
    accounted = success_count + failed_count + skipped_count
    if accounted < total_requested:
        failed_count += total_requested - accounted
    failed_count = min(failed_count, total_requested)
    success_count = max(total_requested - failed_count - skipped_count, 0)

    savings_mb = round(total_savings_bytes / (1024 * 1024), 1)
    savings_pct = round((total_savings_bytes / total_original_bytes) * 100, 1) if total_original_bytes > 0 else 0.0

    # Clear batch state on worker now that we're done
    try:
        await _worker_request(service.worker_url, "PUT", "/batch", {})
    except Exception:
        pass

    status = "success"
    if failed_count:
        status = "failed" if success_count == 0 else "partial"

    # Deferred tagging: submit AI tagging for all successful scenes (sequential mode)
    tag_count = 0
    if not tag_in_parallel and _coerce_bool(settings.get("tag_after_reencode"), True):
        children_final = [task_manager.get(cid) for cid in child_ids]
        scenes_to_tag = []
        for child in children_final:
            if child is None:
                continue
            cr = getattr(child, "result", None)
            if isinstance(cr, dict) and cr.get("needs_tagging") and cr.get("target_scene_id"):
                scenes_to_tag.append(cr["target_scene_id"])
        if scenes_to_tag:
            task_manager.emit_progress(task_record, {
                "status_line": f"Queueing AI tagging for {len(scenes_to_tag)} scene(s)...",
            })
            for sid in scenes_to_tag:
                ok = await _chain_ai_tagging(sid, "")
                if ok:
                    tag_count += 1

    message = (
        f"Re-encode complete: {success_count} succeeded, {skipped_count} skipped, "
        f"{failed_count} failed. Total savings: ({savings_mb:,.1f} MB, {savings_pct}% saved)"
    )
    if tag_count > 0:
        message += f", {tag_count} queued for AI tagging"

    return {
        "status": status,
        "message": message,
        "scenes_requested": total_requested,
        "scenes_completed": success_count,
        "scenes_failed": failed_count,
        "scenes_skipped": skipped_count,
        "total_savings_mb": savings_mb,
        "total_savings_pct": savings_pct,
        "spawned": child_ids,
        "count": len(child_ids),
        "held": True,
    }


def _get_child_progress(child) -> float:
    """Extract the last emitted encode progress percent from a child task."""
    progress = getattr(child, "last_progress", None)
    if isinstance(progress, dict):
        return progress.get("percent", 0.0)
    return 0.0


def _get_child_progress_dict(child) -> dict:
    """Extract rich progress data from a child task."""
    progress = getattr(child, "last_progress", None)
    if isinstance(progress, dict):
        return {
            "percent": progress.get("percent", 0.0),
            "fps": progress.get("fps", 0.0),
            "speed": progress.get("speed", ""),
            "filename": progress.get("filename", ""),
        }
    return {"percent": 0.0, "fps": 0.0, "speed": "", "filename": ""}


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------
class ReencodeService(ServiceBase):
    name = "jiwenji_reencode"
    description = "GPU-accelerated H.265 (HEVC) re-encoding via NVENC"
    max_concurrency = 2  # safe default (1 engine + 1 controller), updated in reload_settings

    def __init__(self) -> None:
        super().__init__()
        self._settings_cache: dict = {}
        self.worker_url: str = _DEFAULT_WORKER_URL
        self.reload_settings()
        # Recover uncollected jobs from a previous backend crash (fire-and-forget)
        try:
            asyncio.get_event_loop().create_task(self._startup_recovery())
        except RuntimeError:
            pass  # no event loop yet — recovery will happen on first batch

    async def _startup_recovery(self) -> None:
        """One-shot recovery: reconnect to any in-flight or uncollected worker jobs."""
        await asyncio.sleep(5)  # give the worker a moment to be reachable
        try:
            await _recover_worker_jobs(self)
        except Exception as exc:
            _log.debug("Startup recovery skipped: %s", exc)

    def reload_settings(self) -> None:
        cfg = self._load_settings()
        self._settings_cache = cfg

        # worker_url lives inside the reencode_advanced blob
        adv = cfg.get("reencode_advanced") or {}
        if not isinstance(adv, dict):
            adv = {}
        self.worker_url = adv.get("worker_url") or _DEFAULT_WORKER_URL

        max_enc = _coerce_int(adv.get("max_concurrent_encodes"), -1)
        if max_enc > 0:
            # +1 for the parent controller task which polls but doesn't encode
            self.max_concurrency = max_enc + 1
        else:
            # Auto-detect: query worker for GPU encoder engine count, fallback to 1
            detected = 1
            try:
                import urllib.request
                import json as _json
                url = f"{self.worker_url.rstrip('/')}/health"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    data = _json.loads(resp.read())
                    engines = data.get("nvenc_engines", 0)
                    if engines > 0:
                        detected = engines
            except Exception:
                pass
            # +1 for the parent controller task which polls but doesn't encode
            self.max_concurrency = detected + 1
            _log.info("Auto-detected max concurrency: %d (+1 controller = %d)", detected, detected + 1)

    def connectivity(self) -> str:
        """Check if the reencode worker sidecar is reachable."""
        try:
            import urllib.request
            import json as _json
            url = f"{self.worker_url.rstrip('/')}/health"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = _json.loads(resp.read())
                if data.get("status") == "ready":
                    gpu = data.get("gpu", "unknown")
                    return f"ready (GPU: {gpu})"
                return data.get("status", "unknown")
        except Exception as exc:
            return f"worker unreachable at {self.worker_url}: {exc}"

    def get_encode_settings(self) -> dict:
        """Build a settings dict for the worker's encoder.

        Always reads fresh from DB so UI changes take effect immediately
        without requiring a restart.
        """
        cfg = self._load_settings()

        # Advanced settings are stored as a JSON object inside reencode_advanced
        adv = cfg.get("reencode_advanced") or {}
        if not isinstance(adv, dict):
            adv = {}

        return {
            # Basic (top-level) settings
            "delete_after_convert": _coerce_bool(cfg.get("delete_after_convert"), True),
            "tag_on_failure": _coerce_bool(cfg.get("tag_on_failure"), True),
            "reencode_failed_tag": cfg.get("reencode_failed_tag") or "reencode_failed",
            # Advanced (from reencode_advanced blob)
            "max_concurrent_encodes": _coerce_int(adv.get("max_concurrent_encodes"), -1),
            "cq": _coerce_int(adv.get("cq"), 28),
            "cq_low_bitrate": _coerce_int(adv.get("cq_low_bitrate"), 34),
            "preset": adv.get("preset") or "p7",
            "skip_codecs": _coerce_skip_codecs(adv),
            "output_suffix": adv.get("output_suffix") or "",
            "min_savings_pct": _coerce_float(adv.get("min_savings_pct"), 15.0),
            "gpu_index": _coerce_int(adv.get("gpu_index"), 0),
            "enable_retries": _coerce_bool(adv.get("enable_retries"), True),
            "aggressive_cq": _coerce_int(adv.get("aggressive_cq"), 34),
            "ultra_aggressive_cq": _coerce_int(adv.get("ultra_aggressive_cq"), 40),
            "skip_failed_tag": _coerce_bool(adv.get("skip_failed_tag"), True),
            "skip_incompatible_container": _coerce_bool(adv.get("skip_incompatible_container"), False),
            "copy_metadata_on_suffix": _coerce_bool(adv.get("copy_metadata_on_suffix"), True),
            "tag_after_reencode": _coerce_bool(cfg.get("tag_after_reencode") if cfg.get("tag_after_reencode") is not None else adv.get("tag_after_reencode"), True),
            "tag_in_parallel": _coerce_bool(adv.get("tag_in_parallel"), True),
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @action(
        id="jiwenji.reencode.scene",
        label="Re-encode to H.265",
        description="Re-encode this scene to H.265 (HEVC) using GPU acceleration",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="single")],
    )
    async def reencode_single(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await reencode_scenes(self, ctx, params, task_record)

    @action(
        id="jiwenji.reencode.scene.selected",
        label="Re-encode Selected",
        description="Re-encode selected scenes to H.265",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="multi")],
    )
    async def reencode_selected(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await reencode_scenes(self, ctx, params, task_record)

    @action(
        id="jiwenji.reencode.scene.page",
        label="Re-encode Page",
        description="Re-encode all scenes on the current page to H.265",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="page")],
    )
    async def reencode_page(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        return await reencode_scenes(self, ctx, params, task_record)

    @action(
        id="jiwenji.reencode.scene.all",
        label="Re-encode All Scenes",
        description="Re-encode every scene in the library to H.265",
        result_kind="dialog",
        contexts=[ContextRule(pages=["scenes"], selection="none")],
    )
    async def reencode_all(self, ctx: ContextInput, params: dict, task_record: TaskRecord):
        ctx.selected_ids = await stash_api.get_all_scenes_async()
        return await reencode_scenes(self, ctx, params, task_record)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

# Keys that are defined in plugin.yml — everything else is orphaned from
# the old per-key storage approach and should be cleaned up.
_VALID_SETTING_KEYS = frozenset({
    "delete_after_convert",
    "tag_on_failure", "reencode_failed_tag", "tag_after_reencode", "reencode_advanced",
})


def _cleanup_orphan_settings():
    """Remove stale rows and ensure correct ordering.

    Deletes orphan rows (keys not in _VALID_SETTING_KEYS) and also detects
    when valid rows are in the wrong order.  When reordering is needed, all
    rows are deleted so register_settings() recreates them from plugin.yml
    in the correct order.  User-set values for the reencode_advanced blob
    are preserved through the reset.
    """
    try:
        from stash_ai_server.db.session import get_session_local
        from stash_ai_server.models.plugin import PluginSetting
        from sqlalchemy import select as sa_select

        # The desired key order matches plugin.yml
        _DESIRED_ORDER = ["tag_on_failure", "reencode_failed_tag", "delete_after_convert", "tag_after_reencode", "reencode_advanced"]

        Session = get_session_local()
        db = Session()
        try:
            rows = db.execute(
                sa_select(PluginSetting).where(PluginSetting.plugin_name == "jiwenji_reencode")
                .order_by(PluginSetting.id)
            ).scalars().all()

            orphans = [r for r in rows if r.key not in _VALID_SETTING_KEYS]
            valid_keys_in_order = [r.key for r in rows if r.key in _VALID_SETTING_KEYS]
            needs_reorder = valid_keys_in_order != _DESIRED_ORDER

            if not orphans and not needs_reorder:
                return

            if needs_reorder:
                # Save user values before nuking
                saved = {}
                for r in rows:
                    if r.key in _VALID_SETTING_KEYS and r.value is not None and r.value != r.default_value:
                        saved[r.key] = r.value
                for r in rows:
                    db.delete(r)
                db.commit()
                _log.info(
                    "Deleted all jiwenji_reencode settings for reorder (had: %s, want: %s, saved %d values)",
                    valid_keys_in_order, _DESIRED_ORDER, len(saved),
                )
                # register_settings() already ran and won't run again, so we
                # need to recreate rows ourselves in the desired order.
                _YML_DEFS = {
                    "tag_on_failure": {"type": "boolean", "label": "Add Tag Indicating Conversion Failed", "default": True, "desc": "Add a tag to scenes that fail re-encoding"},
                    "reencode_failed_tag": {"type": "string", "label": "Failure Tag Name", "default": "reencode_failed", "desc": "Name of the tag to apply when re-encoding fails"},
                    "delete_after_convert": {"type": "boolean", "label": "Delete Original After Successful Re-encode", "default": True, "desc": "Delete the original file after a successful re-encode"},
                    "tag_after_reencode": {"type": "boolean", "label": "Start Skier AI Tagging Job After Successful Re-encode - Requires Skier AI Tagging Plugin", "default": True, "desc": "Automatically run AI tagging after a successful re-encode"},
                    "reencode_advanced": {"type": "reencode_settings", "label": "Advanced Settings", "default": None, "desc": "Configure quality, concurrency, retry, GPU, and post-encode options"},
                }
                for key in _DESIRED_ORDER:
                    d = _YML_DEFS[key]
                    val = saved.get(key, d["default"])
                    row = PluginSetting(
                        plugin_name="jiwenji_reencode",
                        key=key,
                        type=d["type"],
                        label=d["label"],
                        default_value=d["default"],
                        description=d["desc"],
                        value=val,
                    )
                    db.add(row)
                db.commit()
            else:
                # Just orphan cleanup
                for r in orphans:
                    db.delete(r)
                db.commit()
                _log.info("Cleaned up %d orphan setting rows", len(orphans))
        finally:
            db.close()
    except Exception as exc:
        _log.debug("Orphan settings cleanup skipped: %s", exc)


def _fix_setting_types():
    """One-time fix: ensure setting rows have correct types from plugin.yml.

    This repairs damage from a previous cleanup that re-inserted rows with
    type='string' after register_settings had already set the correct types.
    """
    _TYPE_MAP = {
        "tag_on_failure": "boolean",
        "reencode_failed_tag": "string",
        "delete_after_convert": "boolean",
        "tag_after_reencode": "boolean",
        "reencode_advanced": "reencode_settings",
    }
    try:
        from stash_ai_server.db.session import get_session_local
        from stash_ai_server.models.plugin import PluginSetting
        from sqlalchemy import select as sa_select

        Session = get_session_local()
        db = Session()
        try:
            rows = db.execute(
                sa_select(PluginSetting).where(PluginSetting.plugin_name == "jiwenji_reencode")
            ).scalars().all()
            changed = False
            for r in rows:
                expected = _TYPE_MAP.get(r.key)
                if expected and r.type != expected:
                    r.type = expected
                    changed = True
            if changed:
                db.commit()
                _log.info("Fixed setting types for jiwenji_reencode")
        finally:
            db.close()
    except Exception as exc:
        _log.debug("Setting type fix skipped: %s", exc)


def register():
    _cleanup_orphan_settings()
    _fix_setting_types()
    services.register(ReencodeService())
