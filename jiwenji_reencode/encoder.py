from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from jiwenji_reencode import CODEC_FAMILIES

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU engine counts — max simultaneous NVENC sessions per GPU model
# ---------------------------------------------------------------------------
NVENC_ENGINE_COUNTS: dict[str, int] = {
    # Source: https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
    # ── GeForce Consumer ──
    # RTX 50-series (Blackwell)
    "RTX 5090": 3, "RTX 5080": 2, "RTX 5070 Ti": 2,
    "RTX 5070": 1, "RTX 5060 Ti": 1, "RTX 5060": 1,
    # RTX 40-series (Ada Lovelace) — 2 NVENC engines
    "RTX 4090": 2, "RTX 4080 SUPER": 2, "RTX 4080": 2,
    "RTX 4070 Ti SUPER": 2, "RTX 4070 Ti": 2, "RTX 4070 SUPER": 2,
    "RTX 4070": 2, "RTX 4060 Ti": 2, "RTX 4060": 2,
    # RTX 30-series (Ampere) — 1 NVENC engine
    "RTX 3090 Ti": 1, "RTX 3090": 1, "RTX 3080 Ti": 1, "RTX 3080": 1,
    "RTX 3070 Ti": 1, "RTX 3070": 1, "RTX 3060 Ti": 1, "RTX 3060": 1,
    # RTX 20-series (Turing) — 1 NVENC engine
    "RTX 2080 Ti": 1, "RTX 2080 SUPER": 1, "RTX 2080": 1,
    "RTX 2070 SUPER": 1, "RTX 2070": 1, "RTX 2060 SUPER": 1, "RTX 2060": 1,
    # GTX 16-series (Turing) — 1 NVENC engine
    "GTX 1660 Ti": 1, "GTX 1660 SUPER": 1, "GTX 1660": 1, "GTX 1650": 1,
    # GTX 10-series (Pascal) — 1 NVENC engine
    "GTX 1080 Ti": 1, "GTX 1080": 1, "GTX 1070 Ti": 1, "GTX 1070": 1,
    "GTX 1060": 1, "GTX 1050 Ti": 1, "GTX 1050": 1,
    # ── Workstation / Enterprise ──
    # RTX PRO (Blackwell) — 3 NVENC engines (GB202, same die as RTX 5090)
    "RTX PRO 6000": 3,
    # L-series (Ada) — 3 NVENC engines
    "L40S": 3, "L40": 3,
    # RTX Ada workstation
    "RTX 6000 Ada": 3, "RTX 5000 Ada": 2,
    "RTX 4500 Ada": 2, "RTX 4000 Ada": 2,
    # RTX A-series (Ampere)
    "RTX A6000": 1, "RTX A5000": 1,
    "RTX A4500": 1, "RTX A4000": 2, "RTX A2000": 1,
    # A40 (Ampere viz) — 1 NVENC engine
    "A40": 1,
    # Quadro RTX (Turing) — 1 NVENC engine
    "Quadro RTX 8000": 1, "Quadro RTX 6000": 1,
    "Quadro RTX 5000": 1, "Quadro RTX 4000": 1,
    # Quadro Pascal
    "Quadro P6000": 2, "Quadro P5000": 2, "Quadro P4000": 1,
    # Quadro Volta — 3 NVENC engines
    "Quadro GV100": 3,
    # No NVENC: A100, H100, H200, B100, B200, Quadro GP100
}

# ---------------------------------------------------------------------------
# Low-bitrate thresholds — below these, the file is already compact
# ---------------------------------------------------------------------------
LOW_BITRATE_THRESHOLDS: dict[str, int] = {
    "4320p": 40_000_000,   # 8K
    "2160p": 20_000_000,   # 4K
    "1440p": 10_000_000,   # 2K / QHD
    "1080p": 5_000_000,    # FHD
    "720p":  2_500_000,    # HD
    "480p":  1_200_000,    # SD
    "360p":  800_000,
    "240p":  500_000,
    "144p":  300_000,
}

# ---------------------------------------------------------------------------
# Container format mapping for ffmpeg -f flag
# ---------------------------------------------------------------------------
FORMAT_MAP: dict[str, str] = {
    ".mp4": "mp4",
    ".mkv": "matroska",
    ".avi": "avi",
    ".mov": "mov",
    ".wmv": "asf",
    ".flv": "flv",
    ".webm": "matroska",
    ".ts": "mpegts",
    ".m4v": "mp4",
    ".mpg": "mpeg",
    ".mpeg": "mpeg",
    ".3gp": "3gp",
}

# Codecs where CUDA/CUVID hardware decoding is known to silently produce
# corrupt output (green frames) on modern GPUs instead of failing cleanly.
# For these, skip straight to software decoding.
_HWACCEL_BROKEN_CODECS: frozenset[str] = frozenset({
    "vc1",
    "wmv3",
    "wmv2",
    "wmv1",
    "msmpeg4v3",
    "msmpeg4v2",
    "msmpeg4v1",
})

# Containers that cannot hold HEVC — output will be forced to MP4
_HEVC_INCOMPATIBLE_FORMATS: frozenset[str] = frozenset({
    "asf",   # WMV/ASF
    "avi",   # AVI
    "flv",   # FLV
    "3gp",   # 3GP
    "mpeg",  # MPEG-PS
})


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class EncodeResult:
    success: bool
    skipped: bool = False
    skip_reason: str | None = None
    original_size: int = 0
    new_size: int | None = None
    savings_pct: float | None = None
    method_used: str | None = None
    error: str | None = None
    output_path: str | None = None  # set when container format changed (e.g. .wmv → .mp4)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------
async def ffprobe_json(path: str | Path) -> dict:
    """Run ffprobe and return parsed JSON."""
    def _run():
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
        return json.loads(result.stdout)
    return await asyncio.to_thread(_run)


@dataclass
class VideoInfo:
    codec: str
    bitrate: int
    width: int
    height: int
    is_hevc: bool
    duration: float
    pix_fmt: str


async def get_video_info(path: str | Path) -> VideoInfo:
    """Extract video stream info via ffprobe."""
    data = await ffprobe_json(path)
    video_stream = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
            break
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    codec = (video_stream.get("codec_name") or "").lower()
    is_hevc = codec in ("hevc", "h265", "h.265")

    # Bitrate: try stream, then format-level
    bitrate = 0
    if video_stream.get("bit_rate"):
        bitrate = int(video_stream["bit_rate"])
    elif data.get("format", {}).get("bit_rate"):
        bitrate = int(data["format"]["bit_rate"])

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    duration = 0.0
    if video_stream.get("duration"):
        duration = float(video_stream["duration"])
    elif data.get("format", {}).get("duration"):
        duration = float(data["format"]["duration"])

    pix_fmt = video_stream.get("pix_fmt", "yuv420p")

    return VideoInfo(
        codec=codec,
        bitrate=bitrate,
        width=width,
        height=height,
        is_hevc=is_hevc,
        duration=duration,
        pix_fmt=pix_fmt,
    )


def _resolution_label(w: int, h: int) -> str:
    """Map height to resolution label."""
    if h >= 4320:
        return "4320p"
    if h >= 2160:
        return "2160p"
    if h >= 1440:
        return "1440p"
    if h >= 1080:
        return "1080p"
    if h >= 720:
        return "720p"
    if h >= 480:
        return "480p"
    if h >= 360:
        return "360p"
    if h >= 240:
        return "240p"
    return "144p"


def looks_too_low_bitrate(w: int, h: int, bps: int) -> bool:
    """Check if the bitrate is already quite low for this resolution."""
    label = _resolution_label(w, h)
    threshold = LOW_BITRATE_THRESHOLDS.get(label, 1_000_000)
    return bps <= threshold


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
async def detect_gpu_info(gpu_index: int = 0) -> tuple[str, int]:
    """Query nvidia-smi for GPU name and return (gpu_name, engine_count)."""
    def _run():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits",
                 f"--id={gpu_index}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return "Unknown", 1
            gpu_name = result.stdout.strip().split("\n")[0].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return "Unknown", 1

        # Match against known GPU names
        engine_count = 1  # safe default — assume 1 engine for unknown GPUs
        for known_name, count in NVENC_ENGINE_COUNTS.items():
            if known_name.lower() in gpu_name.lower():
                engine_count = count
                break
        return gpu_name, engine_count

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Encode method builders
# ---------------------------------------------------------------------------
@dataclass
class EncodeMethod:
    name: str
    codec: str
    args: list[str]
    min_savings_pct: float = 0.0


def build_nvenc_main10_method(cq: int, preset: str) -> EncodeMethod:
    return EncodeMethod(
        name=f"hevc_nvenc_main10_cq{cq}",
        codec="hevc_nvenc",
        args=[
            "-profile:v", "main10",
            "-rc", "constqp",
            "-qp", str(cq),
            "-preset", preset,
            "-tier", "high",
            "-rc-lookahead", "32",
            "-spatial_aq", "1",
            "-aq-strength", "8",
            "-b:v", "0",
        ],
    )


def build_nvenc_main8_method(cq: int, preset: str) -> EncodeMethod:
    return EncodeMethod(
        name=f"hevc_nvenc_main_cq{cq}",
        codec="hevc_nvenc",
        args=[
            "-profile:v", "main",
            "-rc", "constqp",
            "-qp", str(cq),
            "-preset", preset,
            "-tier", "high",
            "-rc-lookahead", "32",
            "-spatial_aq", "1",
            "-aq-strength", "8",
            "-b:v", "0",
        ],
    )


def build_methods(is_low_bitrate: bool, settings: dict) -> list[EncodeMethod]:
    """Build an ordered list of encode methods to try."""
    cq = settings.get("cq_low_bitrate", 34) if is_low_bitrate else settings.get("cq", 28)
    preset = settings.get("preset", "p7")
    min_savings = settings.get("min_savings_pct", 15)

    methods: list[EncodeMethod] = []

    # Primary: main10 profile
    m10 = build_nvenc_main10_method(cq, preset)
    m10.min_savings_pct = min_savings
    methods.append(m10)

    # Fallback: main (8-bit) profile
    m8 = build_nvenc_main8_method(cq, preset)
    m8.min_savings_pct = min_savings
    methods.append(m8)

    # Retry methods (if enabled)
    if settings.get("enable_retries", True):
        aggressive_cq = settings.get("aggressive_cq", 34)
        if aggressive_cq != cq:
            agg = build_nvenc_main10_method(aggressive_cq, preset)
            agg.name = f"hevc_nvenc_main10_aggressive_cq{aggressive_cq}"
            agg.min_savings_pct = 0.0  # accept any savings on retries
            methods.append(agg)

        ultra_ceiling = settings.get("ultra_aggressive_cq", 40)
        for ucq in range(36, ultra_ceiling + 1, 2):
            if ucq <= aggressive_cq:
                continue
            ultra = build_nvenc_main10_method(ucq, preset)
            ultra.name = f"hevc_nvenc_main10_ultra_cq{ucq}"
            ultra.min_savings_pct = 0.0
            methods.append(ultra)

    return methods


def build_encode_cmd(
    inp: str | Path,
    outp: str | Path,
    method: EncodeMethod,
    fmt: str,
    gpu_idx: int = 0,
    hwaccel: bool = True,
    transcode_audio: bool = False,
    strip_metadata: bool = False,
) -> list[str]:
    """Build the ffmpeg command-line argument list."""
    cmd = ["ffmpeg", "-y"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_idx)]
    cmd += [
        "-i", str(inp),
        "-c:v", method.codec,
        *method.args,
        "-gpu", str(gpu_idx),
    ]
    # When changing container format, the original audio codec may not be
    # compatible (e.g. WMA → MP4).  Transcode to AAC in that case.
    if transcode_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c:a", "copy"]
    # Strip all container/stream metadata (title, website, encoder, etc.)
    if strip_metadata:
        cmd += ["-map_metadata", "-1"]
    cmd += [
        "-map", "0:V",       # capital V excludes attached-pic streams (cover art, thumbnails)
        "-map", "0:a?",
        "-f", fmt,
        "-progress", "pipe:1",
        "-nostats",
        str(outp),
    ]
    return cmd


# ---------------------------------------------------------------------------
# Progress parsing
# ---------------------------------------------------------------------------
async def _read_progress(
    proc: asyncio.subprocess.Process,
    total_duration: float,
    progress_callback: Callable | None,
    cancel_check: Callable | None,
) -> bool:
    """Read ffmpeg -progress pipe:1 output and report percent.
    Returns True if completed, False if cancelled."""
    STALL_TIMEOUT = 120  # seconds with no ffmpeg output at all

    if proc.stdout is None:
        return True

    # Accumulate fields between progress= lines
    cur_fps: float = 0.0
    cur_speed: str = ""

    while True:
        if cancel_check and cancel_check():
            proc.kill()
            return False

        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=STALL_TIMEOUT)
        except asyncio.TimeoutError:
            _log.warning("ffmpeg stalled — no output for %ds, killing process", STALL_TIMEOUT)
            proc.kill()
            return False
        if not line:
            break
        decoded = line.decode("utf-8", errors="replace").strip()
        if decoded.startswith("fps="):
            try:
                cur_fps = float(decoded.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
        elif decoded.startswith("speed="):
            cur_speed = decoded.split("=", 1)[1].strip()
        elif decoded.startswith("out_time_us="):
            try:
                time_us = int(decoded.split("=", 1)[1])
                if total_duration > 0 and progress_callback:
                    pct = min(time_us / (total_duration * 1_000_000) * 100, 100.0)
                    progress_callback({"percent": pct, "fps": cur_fps, "speed": cur_speed})
            except (ValueError, ZeroDivisionError):
                pass
        elif decoded.startswith("progress=end"):
            break

    return True


# ---------------------------------------------------------------------------
# Metadata-only remux (no re-encode)
# ---------------------------------------------------------------------------
async def _metadata_strip_remux(
    input_path: Path,
    info: VideoInfo,
    original_size: int,
    settings: dict,
    progress_callback: Callable | None = None,
    cancel_check: Callable | None = None,
) -> EncodeResult:
    """Remux a file with -map_metadata -1, copying all streams without re-encoding."""
    # Check if the file actually has metadata worth stripping
    try:
        probe = await ffprobe_json(input_path)
        tags = probe.get("format", {}).get("tags", {})
        # Ignore harmless structural tags that ffmpeg always writes
        _IGNORE_TAGS = {"major_brand", "minor_version", "compatible_brands", "encoder"}
        meaningful = {k: v for k, v in tags.items() if k.lower() not in _IGNORE_TAGS}
        if not meaningful:
            _log.info("No meaningful metadata to strip from %s, skipping remux", input_path.name)
            return EncodeResult(
                success=True, skipped=True,
                skip_reason="Already target codec, no metadata to strip",
                original_size=original_size,
            )
        _log.info("Metadata to strip: %s", list(meaningful.keys()))
    except Exception:
        pass  # If probe fails, proceed with remux anyway

    ext = input_path.suffix.lower()
    fmt = FORMAT_MAP.get(ext, "mp4")

    _TEMP_SUFFIX = ".tmp"
    temp_path = input_path.with_name(input_path.name + _TEMP_SUFFIX)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c", "copy",
        "-map_metadata", "-1",
        "-map", "0:V",
        "-map", "0:a?",
        "-f", fmt,
        "-progress", "pipe:1",
        "-nostats",
        str(temp_path),
    ]
    _log.info("Metadata-strip remux: %s", " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        completed = await _read_progress(proc, info.duration, progress_callback, cancel_check)
        if not completed:
            _cleanup_temp(temp_path)
            return EncodeResult(success=False, error="Cancelled")

        _, stderr_data = await proc.communicate()
        if proc.returncode != 0:
            stderr_text = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
            _log.warning("Metadata-strip remux failed: %s", stderr_text[-300:])
            _cleanup_temp(temp_path)
            return EncodeResult(success=False, original_size=original_size, error=f"Metadata remux failed (ffmpeg exit {proc.returncode})")
    except Exception as exc:
        _cleanup_temp(temp_path)
        return EncodeResult(success=False, original_size=original_size, error=f"Metadata remux exception: {exc}")

    if not temp_path.exists() or temp_path.stat().st_size == 0:
        _cleanup_temp(temp_path)
        return EncodeResult(success=False, original_size=original_size, error="Metadata remux produced empty output")

    new_size = temp_path.stat().st_size

    try:
        output_path = _resolve_output_path(input_path, settings)
        _finalize_output(input_path, temp_path, output_path, settings)
    except Exception as exc:
        _cleanup_temp(temp_path)
        return EncodeResult(success=False, error=f"File finalization failed: {exc}")

    savings_pct = ((original_size - new_size) / original_size) * 100 if original_size > 0 else 0.0
    return EncodeResult(
        success=True,
        original_size=original_size,
        new_size=new_size,
        savings_pct=round(savings_pct, 1),
        method_used="metadata_strip_remux",
    )


# ---------------------------------------------------------------------------
# Main encode function
# ---------------------------------------------------------------------------
async def reencode_file(
    input_path: Path,
    settings: dict,
    progress_callback: Callable | None = None,
    cancel_check: Callable | None = None,
) -> EncodeResult:
    """Re-encode a single file to H.265 using NVENC.

    Args:
        input_path: Path to the source video file.
        settings: Dict of plugin settings (cq, preset, etc.).
        progress_callback: Optional callable(pct: float) for progress updates.
        cancel_check: Optional callable() -> bool, returns True to abort.

    Returns:
        EncodeResult with outcome details.
    """
    if not input_path.exists():
        return EncodeResult(success=False, error=f"File not found: {input_path}")

    original_size = input_path.stat().st_size
    if original_size == 0:
        return EncodeResult(success=False, error="File is empty")

    # Probe video info
    try:
        info = await get_video_info(input_path)
    except Exception as exc:
        return EncodeResult(success=False, error=f"ffprobe failed: {exc}")

    # Skip if codec matches any selected skip family
    # (unless strip_metadata is enabled — then do a metadata-only remux)
    strip_metadata = settings.get("strip_metadata", False)
    skip_codecs = settings.get("skip_codecs") or []
    if skip_codecs:
        codec_lower = info.codec.lower()
        for family_key in skip_codecs:
            if codec_lower in CODEC_FAMILIES.get(family_key, frozenset()):
                if strip_metadata:
                    # Already target codec but metadata strip requested —
                    # do a fast copy-remux instead of full re-encode
                    _log.info("Already %s but strip_metadata enabled — doing metadata-only remux", family_key.upper())
                    return await _metadata_strip_remux(
                        input_path, info, original_size, settings,
                        progress_callback, cancel_check,
                    )
                return EncodeResult(
                    success=True, skipped=True,
                    skip_reason=f"Already {family_key.upper()}",
                    original_size=original_size,
                )

    # Determine low bitrate
    is_low_bitrate = looks_too_low_bitrate(info.width, info.height, info.bitrate)

    # Build methods
    methods = build_methods(is_low_bitrate, settings)

    # Determine output format — force MP4 for containers that can't hold HEVC
    ext = input_path.suffix.lower()
    fmt = FORMAT_MAP.get(ext, "mp4")
    format_changed = False
    if fmt in _HEVC_INCOMPATIBLE_FORMATS:
        if not settings.get("remux_incompatible_container", True):
            return EncodeResult(
                success=True, skipped=True,
                skip_reason=f"Container {ext} cannot hold HEVC (remux disabled)",
                original_size=original_size,
            )
        _log.info("Container %r cannot hold HEVC; will output as MP4", fmt)
        fmt = "mp4"
        format_changed = True

    gpu_idx = settings.get("gpu_index", 0)

    # Temp output path — distinctive suffix so cleanup never touches other apps' files
    _TEMP_SUFFIX = ".tmp"
    if format_changed:
        temp_path = input_path.with_suffix(".mp4" + _TEMP_SUFFIX)
    else:
        temp_path = input_path.with_name(input_path.name + _TEMP_SUFFIX)

    # Try with hardware-accelerated decoding first, then fall back to
    # software decoding.  CUVID doesn't support every input codec (e.g.
    # WMV3, some MPEG-4 ASP variants), so the fallback is essential.
    # For codecs known to produce silent corruption with CUDA, skip
    # straight to software decode.
    new_ext = ".mp4" if format_changed else None
    skip_hwaccel = info.codec in _HWACCEL_BROKEN_CODECS
    if skip_hwaccel:
        _log.info("Codec %r is in HWACCEL_BROKEN_CODECS — skipping CUDA decode", info.codec)

    for hwaccel in (False,) if skip_hwaccel else (True, False):
        result = await _try_methods(
            input_path, temp_path, methods, fmt, gpu_idx, hwaccel,
            info.duration, original_size, settings,
            progress_callback, cancel_check, new_ext=new_ext,
            force_audio_transcode=format_changed,
        )
        if result is not None:
            return result
        if not hwaccel:
            break
        _log.info("All methods failed with hwaccel cuda; retrying with software decode")

    # All methods exhausted (both hwaccel passes)
    _cleanup_temp(temp_path)
    return EncodeResult(
        success=False,
        original_size=original_size,
        error="All encode methods failed (tried both hw and sw decode)",
    )


def _looks_like_audio_compat_error(stderr_text: str) -> bool:
    """Check if ffmpeg stderr suggests an audio codec/container incompatibility."""
    lower = stderr_text.lower()
    return any(phrase in lower for phrase in (
        "no packets",
        "could not find tag for codec",
        "not currently supported in container",
        "unsupported codec",
        "codec not currently supported",
        "tag not found",
    ))


async def _try_methods(
    input_path: Path,
    temp_path: Path,
    methods: list[EncodeMethod],
    fmt: str,
    gpu_idx: int,
    hwaccel: bool,
    duration: float,
    original_size: int,
    settings: dict,
    progress_callback: Callable | None,
    cancel_check: Callable | None,
    new_ext: str | None = None,
    force_audio_transcode: bool = False,
) -> EncodeResult | None:
    """Try all encode methods with the given hwaccel mode.

    Returns an EncodeResult on success/cancel, or None if all methods failed
    and the caller should try the next hwaccel mode.
    """
    decode_label = "hwaccel cuda" if hwaccel else "software decode"
    all_instant_failures = True  # Track if every method failed with 0 frames

    for method in methods:
        if cancel_check and cancel_check():
            _cleanup_temp(temp_path)
            return EncodeResult(success=False, error="Cancelled")

        # When the container format changed (e.g. WMV→MP4), the original
        # audio codec (e.g. WMA) almost certainly can't be copied into the
        # new container — skip straight to AAC transcoding.
        audio_options = (True,) if force_audio_transcode else (False, True)
        for audio_transcode in audio_options:
            cmd = build_encode_cmd(input_path, temp_path, method, fmt, gpu_idx, hwaccel=hwaccel, transcode_audio=audio_transcode, strip_metadata=settings.get("strip_metadata", False))
            audio_label = " +aac" if audio_transcode else ""
            _log.info("Trying method %s (%s%s): %s", method.name, decode_label, audio_label, " ".join(cmd))

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                completed = await _read_progress(proc, duration, progress_callback, cancel_check)
                if not completed:
                    _cleanup_temp(temp_path)
                    return EncodeResult(success=False, error="Cancelled")

                _, stderr_data = await proc.communicate()
                returncode = proc.returncode

            except Exception as exc:
                _log.warning("Method %s raised exception: %s", method.name, exc)
                _cleanup_temp(temp_path)
                break  # move to next method

            if returncode == 0:
                break  # success — fall through to size check below

            stderr_text = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
            _log.warning("Method %s failed: ffmpeg exited %d: %s", method.name, returncode, stderr_text[-300:])
            _cleanup_temp(temp_path)

            if not audio_transcode and _looks_like_audio_compat_error(stderr_text):
                # Likely audio codec incompatibility — retry with AAC
                _log.info("Retrying %s with audio transcode (AAC)", method.name)
                continue
            # If this looks like a real encoding attempt (not an instant
            # hwaccel incompatibility), keep trying methods but don't
            # fall through to the sw-decode retry.
            if "frame=" in stderr_text and "frame=    0" not in stderr_text:
                all_instant_failures = False
            break  # move to next method
        else:
            # audio_transcode loop exhausted without success
            continue

        if returncode != 0:
            continue

        # Check output file exists and is valid
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            _log.warning("Method %s: output file missing or empty", method.name)
            _cleanup_temp(temp_path)
            continue

        # Validate the output is a real, playable video — CUVID can
        # silently produce corrupt green-frame output for unsupported
        # codecs while still exiting 0.
        try:
            out_info = await get_video_info(temp_path)
            if out_info.width == 0 or out_info.height == 0:
                _log.warning("Method %s: output has 0×0 dimensions — corrupt", method.name)
                _cleanup_temp(temp_path)
                continue
            if not out_info.is_hevc:
                _log.warning("Method %s: output codec is %r, expected HEVC — corrupt", method.name, out_info.codec)
                _cleanup_temp(temp_path)
                continue
            if out_info.duration > 0 and duration > 0:
                ratio = out_info.duration / duration
                if ratio < 0.5:
                    _log.warning(
                        "Method %s: output duration %.1fs is much shorter than input %.1fs — likely corrupt",
                        method.name, out_info.duration, duration,
                    )
                    _cleanup_temp(temp_path)
                    continue
        except Exception as exc:
            _log.warning("Method %s: output validation failed (ffprobe error: %s) — treating as corrupt", method.name, exc)
            _cleanup_temp(temp_path)
            continue

        all_instant_failures = False
        new_size = temp_path.stat().st_size
        savings_pct = ((original_size - new_size) / original_size) * 100 if original_size > 0 else 0.0

        # Check minimum savings threshold
        if savings_pct < method.min_savings_pct:
            _log.info(
                "Method %s: savings %.1f%% below threshold %.1f%%, trying next",
                method.name, savings_pct, method.min_savings_pct,
            )
            _cleanup_temp(temp_path)
            continue

        # Success! Handle file placement
        try:
            output_path = _resolve_output_path(input_path, settings, new_ext=new_ext)
            _finalize_output(input_path, temp_path, output_path, settings)
        except Exception as exc:
            _cleanup_temp(temp_path)
            return EncodeResult(success=False, error=f"File finalization failed: {exc}")

        return EncodeResult(
            success=True,
            original_size=original_size,
            new_size=new_size,
            savings_pct=round(savings_pct, 1),
            method_used=method.name,
            output_path=str(output_path) if new_ext else None,
        )

    # All methods failed in this hwaccel mode.
    # If hwaccel=True and every failure was instant (0 frames / Invalid argument),
    # return None so the caller retries with software decode.
    # If hwaccel=False (already the fallback), or some methods actually ran,
    # this is a genuine failure — also return None to let the caller emit the
    # final EncodeResult.
    if hwaccel and all_instant_failures:
        return None  # Signal: retry with software decode
    _cleanup_temp(temp_path)
    return EncodeResult(
        success=False,
        original_size=original_size,
        error="All encode methods failed",
    )


# ---------------------------------------------------------------------------
# File management helpers
# ---------------------------------------------------------------------------
def _cleanup_temp(temp_path: Path) -> None:
    """Remove temp .part file if it exists."""
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError as exc:
        _log.warning("Failed to clean up temp file %s: %s", temp_path, exc)


def _resolve_output_path(input_path: Path, settings: dict, new_ext: str | None = None) -> Path:
    """Determine where the final encoded file should end up."""
    suffix = settings.get("output_suffix", "")
    delete_after = settings.get("delete_after_convert", True)

    if not suffix and not delete_after:
        # Auto-use _hevc suffix when not deleting and no suffix specified
        suffix = "_hevc"

    out_ext = new_ext or input_path.suffix

    if suffix:
        stem = input_path.stem + suffix
        return input_path.with_name(stem + out_ext)
    else:
        # Replace in-place (extension may change if container was incompatible)
        return input_path.with_suffix(out_ext)


def _finalize_output(
    input_path: Path,
    temp_path: Path,
    output_path: Path,
    settings: dict,
) -> None:
    """Move temp file to final location and optionally delete original."""
    delete_after = settings.get("delete_after_convert", True)

    if output_path == input_path:
        # In-place replacement: delete original, rename temp
        input_path.unlink()
        temp_path.rename(output_path)
    else:
        # Suffix mode: rename temp to suffixed path
        temp_path.rename(output_path)
        if delete_after:
            input_path.unlink()
