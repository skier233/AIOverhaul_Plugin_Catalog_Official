"""Lightweight HTTP worker for GPU-accelerated H.265 re-encoding.

Runs inside the reencode_worker container with GPU + rw media access.
The backend plugin (jiwenji_reencode) sends encode requests here.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from pathlib import Path
from aiohttp import web

# Import encoder from the plugin directory (mounted at /app/plugins/jiwenji_reencode)
import sys
sys.path.insert(0, "/app/plugins")
from jiwenji_reencode import encoder

_log = logging.getLogger("reencode_worker")

# Active encode jobs: job_id -> {task, result, progress}
_jobs: dict[str, dict] = {}

# Semaphore limiting concurrent ffmpeg encodes to the GPU's NVENC engine count.
# Initialized on startup after querying the GPU.
_encode_semaphore: asyncio.Semaphore | None = None

# ---------------------------------------------------------------------------
# Path translation — resolve incoming paths to local container paths
# ---------------------------------------------------------------------------
# The worker's media is mounted at /app/stash (from docker-compose volume).
# Stash sends Windows paths like Z:\f\subdir\file.mp4 or plugin mappings
# may partially translate them.  We try, in order:
#   1. Path as-is (already correct / pre-mapped)
#   2. REENCODE_PATH_MAPPINGS env var (JSON list of [source, target] pairs)
#   3. Auto-detect: strip Windows drive prefix, convert backslashes,
#      and check under /app/stash

def _load_env_mappings() -> list[tuple[str, str]]:
    """Load path mappings from REENCODE_PATH_MAPPINGS env var."""
    raw = os.environ.get("REENCODE_PATH_MAPPINGS", "")
    if not raw:
        return []
    try:
        pairs = json.loads(raw)
        return [(str(s), str(t)) for s, t in pairs]
    except Exception:
        _log.warning("Failed to parse REENCODE_PATH_MAPPINGS env var")
        return []

_env_mappings: list[tuple[str, str]] = []

def _resolve_path(file_path: str) -> Path:
    """Translate an incoming file path to a local container path."""
    # 1. Try as-is
    p = Path(file_path)
    if p.exists():
        return p

    # 2. Try explicit env mappings
    global _env_mappings
    if not _env_mappings:
        _env_mappings = _load_env_mappings()
    for source, target in _env_mappings:
        if file_path.lower().startswith(source.lower()):
            remainder = file_path[len(source):]
            candidate = Path(target) / remainder.replace("\\", "/").lstrip("/")
            if candidate.exists():
                return candidate

    # 3. Auto-detect: strip Windows drive letter, map to /app/stash
    #    Z:\f\subdir\file.mp4  →  /app/stash/subdir/file.mp4
    #    Z:\f/subdir/file.mp4  →  /app/stash/subdir/file.mp4
    stash_mount = Path("/app/stash")
    if stash_mount.is_dir():
        # Normalize: replace backslashes, strip drive letter
        normalized = file_path.replace("\\", "/")
        # Strip drive prefix like "Z:/f/" or "Z:/"
        # Try progressively shorter prefixes
        parts = normalized.split("/")
        for i in range(len(parts)):
            remainder = "/".join(parts[i + 1:])
            if not remainder:
                continue
            candidate = stash_mount / remainder
            if candidate.exists():
                _log.info("Auto-resolved path: %s → %s", file_path, candidate)
                return candidate

    # Nothing worked — return original (caller will get "not found")
    return p


async def handle_health(request: web.Request) -> web.Response:
    """Health check — confirms ffmpeg + GPU are available."""
    import shutil
    gpu_name, engine_count = await encoder.detect_gpu_info(0)
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    return web.json_response({
        "status": "ready" if ffmpeg_ok else "ffmpeg_missing",
        "ffmpeg": ffmpeg_ok,
        "gpu": gpu_name,
        "nvenc_engines": engine_count,
    })


async def handle_encode(request: web.Request) -> web.Response:
    """Start an encode job. Returns a job_id for polling progress."""
    body = await request.json()
    file_path = body.get("file_path")
    settings = body.get("settings", {})

    if not file_path:
        return web.json_response({"error": "file_path required"}, status=400)

    path = _resolve_path(file_path)
    if not path.exists():
        return web.json_response({
            "error": f"File not found: {file_path} (resolved to: {path})"
        }, status=404)

    job_id = str(uuid.uuid4())
    job = {
        "progress": 0.0, "result": None, "cancelled": False,
        "last_progress_value": 0.0, "last_progress_time": time.monotonic(),
    }
    _jobs[job_id] = job

    def progress_cb(pct: float):
        rounded = round(pct, 3)
        if rounded != job["progress"]:
            job["last_progress_value"] = rounded
            job["last_progress_time"] = time.monotonic()
        job["progress"] = rounded

    def cancel_cb() -> bool:
        return job["cancelled"]

    async def run_encode():
        try:
            async with _encode_semaphore:
                result = await encoder.reencode_file(path, settings, progress_cb, cancel_cb)
            job["result"] = {
                "success": result.success,
                "skipped": result.skipped,
                "skip_reason": result.skip_reason,
                "original_size": result.original_size,
                "new_size": result.new_size,
                "savings_pct": result.savings_pct,
                "method_used": result.method_used,
                "error": result.error,
                "output_path": result.output_path,
            }
        except Exception as exc:
            _log.exception("Encode job %s failed", job_id)
            job["result"] = {"success": False, "error": str(exc)}

    job["task"] = asyncio.create_task(run_encode())
    return web.json_response({"job_id": job_id})


async def handle_status(request: web.Request) -> web.Response:
    """Poll job progress/result."""
    job_id = request.match_info["job_id"]
    job = _jobs.get(job_id)
    if not job:
        return web.json_response({"error": "Unknown job_id"}, status=404)

    resp = {"job_id": job_id, "progress": job["progress"]}
    if job["result"] is not None:
        resp["done"] = True
        resp["result"] = job["result"]
        # Clean up finished job after returning result
        _jobs.pop(job_id, None)
    else:
        resp["done"] = False
    return web.json_response(resp)


async def handle_cancel(request: web.Request) -> web.Response:
    """Cancel a running job."""
    job_id = request.match_info["job_id"]
    job = _jobs.get(job_id)
    if not job:
        return web.json_response({"error": "Unknown job_id"}, status=404)
    job["cancelled"] = True
    return web.json_response({"job_id": job_id, "cancelled": True})


_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".ts", ".m4v", ".mpg", ".mpeg", ".3gp"}

STALE_PROGRESS_TIMEOUT = 300  # 5 min with zero progress movement


async def _reap_stale_jobs():
    """Background task: cancel jobs whose progress hasn't advanced."""
    while True:
        await asyncio.sleep(60)
        now = time.monotonic()
        for job_id in list(_jobs.keys()):
            job = _jobs.get(job_id)
            if not job or job["result"] is not None:
                continue
            last_change = job.get("last_progress_time", now)
            if now - last_change > STALE_PROGRESS_TIMEOUT:
                _log.warning(
                    "Reaping stale job %s — progress stuck at %.3f%% for %ds",
                    job_id, job["progress"], int(now - last_change),
                )
                job["cancelled"] = True
                task = job.get("task")
                if task and not task.done():
                    task.cancel()
                _jobs.pop(job_id, None)


async def _cleanup_orphaned_parts():
    """On boot, remove .part files left by a previous crash/restart."""
    stash_mount = Path("/app/stash")
    if not stash_mount.is_dir():
        return
    now = time.time()
    max_age = 4 * 86400   # 4 days
    min_age = 60           # skip very recent files
    count = 0
    for part_file in stash_mount.rglob("*.part"):
        # Check it matches our temp naming: somefile.mp4.part
        base = part_file.stem  # e.g. "somefile.mp4" from "somefile.mp4.part"
        base_ext = Path(base).suffix.lower()
        if base_ext not in _VIDEO_EXTENSIONS:
            continue
        try:
            mtime = part_file.stat().st_mtime
        except OSError:
            continue
        age = now - mtime
        if age > max_age or age < min_age:
            continue
        _log.info("Removing orphaned temp file: %s (age: %.0fh)", part_file, age / 3600)
        try:
            part_file.unlink()
            count += 1
        except OSError as exc:
            _log.warning("Failed to remove %s: %s", part_file, exc)
    if count:
        _log.info("Cleaned up %d orphaned .part file(s)", count)


async def on_startup(app: web.Application):
    """Initialize the encode semaphore based on detected GPU engine count."""
    global _encode_semaphore
    gpu_name, engine_count = await encoder.detect_gpu_info(0)
    _encode_semaphore = asyncio.Semaphore(engine_count)
    _log.info("Initialized encode semaphore: %d slots (GPU: %s)", engine_count, gpu_name)
    await _cleanup_orphaned_parts()
    asyncio.create_task(_reap_stale_jobs())


def create_app() -> web.Application:
    app = web.Application()
    app.on_startup.append(on_startup)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/encode", handle_encode)
    app.router.add_get("/status/{job_id}", handle_status)
    app.router.add_post("/cancel/{job_id}", handle_cancel)
    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    port = int(os.environ.get("REENCODE_WORKER_PORT", "4154"))
    _log.info("Starting reencode worker on port %d", port)
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
