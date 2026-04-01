import asyncio
import logging
from typing import Sequence
from pydantic import TypeAdapter
from .models import AIModelInfo, AIVideoResultV3, ImageResult, VideoServerResponse
from stash_ai_server.services.base import RemoteServiceBase

IMAGES_ENDPOINT = "/v3/process_images/"  # Batch endpoint - accepts multiple image paths
SCENE_ENDPOINT = "/v3/process_video/"    # Single scene endpoint - processes one scene at a time
ACTIVE_SCENE_MODELS = "/v3/current_ai_models/"

# Face-scan-only endpoints — detection + embedding without tag classification
FACE_SCAN_IMAGES_ENDPOINT = "/v3/face_recognition/process_images/"
FACE_SCAN_VIDEO_ENDPOINT = "/v3/face_recognition/process_video/"


_log = logging.getLogger(__name__)

async def call_images_api(service: RemoteServiceBase, image_paths: list[str]) -> ImageResult | None:
    """Call the /images endpoint with a batch of image paths."""
    try:
        payload = {
            "paths": image_paths,
            "threshold": 0.5,
            "return_confidence": False
        }
        return await service.http.post(
            IMAGES_ENDPOINT,
            json=payload,
            response_model=ImageResult,
        )
    except asyncio.CancelledError:  # pragma: no cover
        raise
    except Exception as exc:  # noqa: BLE001
        _log.warning("images API call failed: %s", exc)
        raise

async def call_scene_api(
    service: RemoteServiceBase,
    scene_path: str,
    frame_interval: float,
    vr_video: bool,
    *,
    threshold: float,
    skip_categories: Sequence[str] | None = None,
) -> VideoServerResponse | None:
    """Call the /scene endpoint for a single scene."""
    try:
        payload = {
            "path": scene_path,
            "frame_interval": frame_interval,
            "return_confidence": True,
            "vr_video": vr_video,
            "threshold": threshold,
        }
        if skip_categories:
            payload["categories_to_skip"] = list(skip_categories)
        return await service.http.post(
            SCENE_ENDPOINT,
            json=payload,
            response_model=VideoServerResponse
        )
    except asyncio.CancelledError:  # pragma: no cover
        raise
    except Exception as exc:  # noqa: BLE001
        _log.warning("scene API call failed for scene_path=%s: %s", scene_path, exc)
        return None

async def get_active_scene_models(service: RemoteServiceBase) -> list[AIModelInfo]:
    """Fetch the list of active models from the remote service."""
    try:
        return await service.http.get(
            ACTIVE_SCENE_MODELS,
            response_model=TypeAdapter(list[AIModelInfo]),
        )
    except asyncio.CancelledError:  # pragma: no cover
        raise
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to fetch active models: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Face-scan-only endpoints
# ---------------------------------------------------------------------------

async def call_face_scan_images_api(
    service: RemoteServiceBase,
    image_paths: list[str],
) -> ImageResult | None:
    """Call the face-scan-only endpoint for a batch of images.

    Returns the same ``ImageResult`` shape but only face detection +
    embedding data (no tag classification).
    """
    try:
        payload = {
            "paths": image_paths,
        }
        return await service.http.post(
            FACE_SCAN_IMAGES_ENDPOINT,
            json=payload,
            response_model=ImageResult,
        )
    except asyncio.CancelledError:  # pragma: no cover
        raise
    except Exception as exc:  # noqa: BLE001
        _log.warning("face scan images API call failed: %s", exc)
        raise


async def call_face_scan_video_api(
    service: RemoteServiceBase,
    scene_path: str,
    frame_interval: float,
    vr_video: bool = False,
) -> VideoServerResponse | None:
    """Call the face-scan-only endpoint for a single video.

    Returns the same ``VideoServerResponse`` shape but only face detection +
    embedding data (no tag classification / timespans).
    """
    try:
        payload = {
            "path": scene_path,
            "frame_interval": frame_interval,
            "vr_video": vr_video,
        }
        return await service.http.post(
            FACE_SCAN_VIDEO_ENDPOINT,
            json=payload,
            response_model=VideoServerResponse,
        )
    except asyncio.CancelledError:  # pragma: no cover
        raise
    except Exception as exc:  # noqa: BLE001
        _log.warning("face scan video API call failed for scene_path=%s: %s", scene_path, exc)
        return None
