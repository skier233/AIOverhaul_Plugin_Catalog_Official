"""Response parser for v3 AI Model Server API responses.

Classifies and normalises the mixed dynamic-key response format into typed,
structured intermediate representations.  Separates:

* **Tagging** results  (``list[str]`` or ``list[tuple[str, float]]``)
* **Detection** results (bounding boxes with score and detector)
* **Region** results    (sub-model outputs on detected crops, e.g. embeddings)

The ``models`` metadata from the response drives classification; heuristic
fall-backs handle cases where metadata is missing or incomplete.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from .models import (
    AIModelInfo,
    Detection,
    EmbeddingResult,
    ParsedFrameData,
    ParsedImageData,
    ParsedVideoData,
    RegionResult,
    TagTimeFrame,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capability constants (values from the model server's YAML config)
# ---------------------------------------------------------------------------

CAPABILITY_TAGGING = "tagging"
CAPABILITY_DETECTION = "detection"
CAPABILITY_EMBEDDING = "embedding"

_REGION_PREFIX = "regions__"

# ---------------------------------------------------------------------------
# Category classifier
# ---------------------------------------------------------------------------


def build_category_classifier(
    models: Sequence[AIModelInfo | Mapping[str, Any]] | None,
) -> dict[str, str]:
    """Build a map from category name to primary capability type.

    Uses the ``capabilities`` field from each model's metadata.

    Returns e.g.::

        {
            "bodyparts": "tagging",
            "face_detections": "detection",
            "face_embeddings": "embedding",
        }
    """
    classifier: dict[str, str] = {}
    if not models:
        return classifier

    for model in models:
        if isinstance(model, Mapping):
            capabilities = model.get("capabilities") or []
            categories = model.get("categories") or []
        else:
            capabilities = model.capabilities or []
            categories = model.categories or []

        # Determine primary capability
        capability = "unknown"
        if CAPABILITY_TAGGING in capabilities:
            capability = CAPABILITY_TAGGING
        elif CAPABILITY_DETECTION in capabilities:
            capability = CAPABILITY_DETECTION
        elif CAPABILITY_EMBEDDING in capabilities:
            capability = CAPABILITY_EMBEDDING
        elif capabilities:
            capability = capabilities[0]

        for cat in categories:
            classifier[cat] = capability

    return classifier


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------


def _is_tag_list(value: Any) -> bool:
    """Return *True* if *value* looks like a tagging output."""
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    if isinstance(first, str):
        return True
    # return_confidence=true produces [label, score] or (label, score) pairs
    if isinstance(first, (list, tuple)) and len(first) == 2:
        return isinstance(first[0], str)
    return False


def _is_detection_list(value: Any) -> bool:
    """Return *True* if *value* looks like a detection output."""
    if not isinstance(value, list) or not value:
        return False
    first = value[0]
    return isinstance(first, dict) and "bbox" in first


# ---------------------------------------------------------------------------
# Low-level parsers
# ---------------------------------------------------------------------------


def _parse_detection(raw: Mapping[str, Any]) -> Detection:
    return Detection(
        bbox=raw.get("bbox", []),
        score=float(raw.get("score", 0.0)),
        detector=raw.get("detector", ""),
    )


def _parse_detections(items: Sequence[Any]) -> list[Detection]:
    detections: list[Detection] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        try:
            detections.append(_parse_detection(raw))
        except (TypeError, ValueError) as exc:
            _log.debug("Skipping malformed detection entry: %s", exc)
    return detections


def _parse_region_entry(entry: Mapping[str, Any]) -> RegionResult:
    detection_index = entry.get("detection_index", 0)
    model_outputs = {k: v for k, v in entry.items() if k != "detection_index"}
    return RegionResult(detection_index=detection_index, model_outputs=model_outputs)


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

_DEFAULT_SKIP_KEYS: frozenset[str] = frozenset({"error"})


def _classify_result_dict(
    data: Mapping[str, Any],
    classifier: dict[str, str],
    *,
    skip_keys: frozenset[str] = _DEFAULT_SKIP_KEYS,
) -> tuple[dict[str, list], dict[str, list[Detection]], dict[str, list[RegionResult]]]:
    """Classify a flat result dict into tags, detections and regions."""
    tags: dict[str, list] = {}
    detections: dict[str, list[Detection]] = {}
    regions: dict[str, list[RegionResult]] = {}

    for key, value in data.items():
        if key in skip_keys:
            continue
        if not isinstance(value, list):
            continue

        # ---- Region keys have a fixed prefix --------------------------
        if key.startswith(_REGION_PREFIX):
            regions[key] = [
                _parse_region_entry(entry)
                for entry in value
                if isinstance(entry, dict)
            ]
            continue

        # ---- Use classifier (model-metadata-driven) -------------------
        cap = classifier.get(key)
        if cap == CAPABILITY_TAGGING:
            tags[key] = value
        elif cap == CAPABILITY_DETECTION:
            detections[key] = _parse_detections(value)
        elif cap == CAPABILITY_EMBEDDING:
            # Top-level embeddings are unusual; log for visibility
            _log.debug("Top-level embedding category '%s'; skipping (not a tag)", key)
        elif cap is not None:
            _log.debug("Unknown capability '%s' for category '%s'", cap, key)
        else:
            # ---- No classifier entry — fall back to heuristics --------
            if _is_tag_list(value):
                tags[key] = value
            elif _is_detection_list(value):
                detections[key] = _parse_detections(value)
                _log.debug(
                    "Heuristic: classified '%s' as detection (missing model metadata)", key
                )
            else:
                _log.debug(
                    "Skipping unclassified key '%s' with %d entries", key, len(value)
                )

    return tags, detections, regions


# ---------------------------------------------------------------------------
# Public parsing API
# ---------------------------------------------------------------------------


def parse_image_result(
    per_image_data: Mapping[str, Any] | Any,
    classifier: dict[str, str],
) -> ParsedImageData:
    """Parse a single per-image result dict into a normalised structure.

    Args:
        per_image_data: One entry from the image response ``result`` array.
        classifier: Output of :func:`build_category_classifier`.

    Returns:
        A :class:`ParsedImageData` with tags, detections and regions separated.
    """
    if not isinstance(per_image_data, Mapping):
        return ParsedImageData(error="unexpected result format")

    error = per_image_data.get("error")
    if error:
        return ParsedImageData(error=str(error))

    tags, detections, regions = _classify_result_dict(per_image_data, classifier)
    return ParsedImageData(tags=tags, detections=detections, regions=regions)


_FRAME_SKIP_KEYS: frozenset[str] = frozenset({"error", "frame_index"})


def parse_frame_data(
    frame_dict: Mapping[str, Any],
    classifier: dict[str, str],
) -> ParsedFrameData:
    """Parse a single video frame entry into a normalised structure."""
    frame_index = float(frame_dict.get("frame_index", 0.0))
    _, detections, regions = _classify_result_dict(
        frame_dict,
        classifier,
        skip_keys=_FRAME_SKIP_KEYS,
    )
    return ParsedFrameData(
        frame_index=frame_index,
        detections=detections,
        regions=regions,
    )


def parse_video_frames(
    raw_frames: Sequence[Mapping[str, Any]] | None,
    classifier: dict[str, str],
) -> list[ParsedFrameData] | None:
    """Parse the ``frames`` array from a video response.

    Returns ``None`` when no structured frame data is present.
    """
    if not raw_frames:
        return None
    parsed: list[ParsedFrameData] = []
    for frame_dict in raw_frames:
        if not isinstance(frame_dict, Mapping):
            continue
        parsed.append(parse_frame_data(frame_dict, classifier))
    return parsed or None


def parse_video_result(
    video_result: Any,
    models: Sequence[AIModelInfo | Mapping[str, Any]] | None = None,
) -> ParsedVideoData:
    """Parse a complete video result into a fully normalised structure.

    Args:
        video_result: An ``AIVideoResultV3`` instance or raw dict.
        models: Optional model list override (falls back to result.models).
    """
    if isinstance(video_result, Mapping):
        result_dict = dict(video_result)
    elif hasattr(video_result, "model_dump"):
        result_dict = video_result.model_dump()
    else:
        result_dict = {}

    model_list = models or result_dict.get("models") or []
    classifier = build_category_classifier(model_list)

    raw_frames = result_dict.get("frames")
    parsed_frames = parse_video_frames(raw_frames, classifier) if raw_frames else None

    # Rebuild models as AIModelInfo when they are dicts
    typed_models: list[AIModelInfo] = []
    for m in model_list:
        if isinstance(m, AIModelInfo):
            typed_models.append(m)
        elif isinstance(m, dict):
            try:
                typed_models.append(AIModelInfo(**m))
            except Exception:
                _log.debug("Skipping malformed model entry: %s", m)

    timespans = result_dict.get("timespans") or {}
    typed_timespans: dict[str, dict[str, list[TagTimeFrame]]] = {}
    for cat, tag_dict in timespans.items():
        if not isinstance(tag_dict, dict):
            continue
        typed_timespans[cat] = {}
        for tag_name, frames in tag_dict.items():
            if isinstance(frames, list):
                typed_timespans[cat][tag_name] = [
                    TagTimeFrame(**f) if isinstance(f, dict) else f
                    for f in frames
                ]

    return ParsedVideoData(
        schema_version=result_dict.get("schema_version", 3),
        duration=float(result_dict.get("duration", 0.0)),
        frame_interval=float(result_dict.get("frame_interval", 0.0)),
        models=typed_models,
        timespans=typed_timespans,
        frames=parsed_frames,
    )


def extract_tags_only(
    per_image_data: Mapping[str, Any],
    classifier: dict[str, str],
) -> dict[str, list]:
    """Extract *only* tagging results from a per-image result dict.

    This is the v3-aware replacement for the older
    ``utils.extract_tags_from_response``.  Detection dicts, region results
    and other non-tag entries are excluded so downstream tag processing does
    not receive dicts where it expects strings.
    """
    tags, _, _ = _classify_result_dict(per_image_data, classifier)
    return tags


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def count_detections(parsed: ParsedImageData | ParsedFrameData) -> int:
    """Total detections across all categories."""
    return sum(len(dets) for dets in parsed.detections.values())


def count_regions(parsed: ParsedImageData | ParsedFrameData) -> int:
    """Total region results across all aliases."""
    return sum(len(regs) for regs in parsed.regions.values())


def parse_embeddings(region: RegionResult, category: str) -> list[EmbeddingResult]:
    """Extract typed embeddings from a region's model outputs.

    Convenience helper for downstream code that needs typed embedding access.

    Args:
        region: A :class:`RegionResult` instance.
        category: The embedding category key (e.g. ``"face_embeddings"``).
    """
    raw = region.model_outputs.get(category) or []
    embeddings: list[EmbeddingResult] = []
    for entry in raw:
        if isinstance(entry, dict):
            try:
                embeddings.append(EmbeddingResult(**entry))
            except (TypeError, ValueError):
                _log.debug("Skipping malformed embedding in category '%s'", category)
    return embeddings
