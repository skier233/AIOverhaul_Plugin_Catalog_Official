
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# API response models — raw shapes returned by the AI model server
# ---------------------------------------------------------------------------


class AIModelInfo(BaseModel):
    """Metadata for a single AI model reported by the model server."""
    name: str
    identifier: int
    version: float
    categories: List[str]
    type: str
    capabilities: List[str] = []
    supported_scopes: List[str] = []


class ImageResult(BaseModel):
    """Top-level response from ``POST /v3/process_images/``."""
    result: List[Dict[str, Any]] = Field(..., min_items=1)
    models: List[AIModelInfo] | None = None
    metrics: Dict[str, Any] | None = None


Scope = Literal["detail", "selected", "page", "all"]


class TagTimeFrame(BaseModel):
    start: float
    end: Optional[float] = None
    confidence: Optional[float] = None
    def __str__(self):
        return f"TagTimeFrame(start={self.start}, end={self.end}, confidence={self.confidence})"


class AIVideoResultV3(BaseModel):
    """The ``result`` object from ``POST /v3/process_video/``."""
    schema_version: int
    duration: float
    models: List[AIModelInfo]
    frame_interval: float
    # category -> tag -> list of timeframes  (tagging data only)
    timespans: Dict[str, Dict[str, List[TagTimeFrame]]]
    # Per-frame structured data: detections, regions, embeddings.
    # Only frames containing non-tag data are included; null when absent.
    frames: List[Dict[str, Any]] | None = None

    def to_json(self):
        return self.model_dump_json(exclude_none=True)


class VideoServerResponse(BaseModel):
    result: AIVideoResultV3 | None = None
    metrics: Dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Parsed / normalised types — output of response_parser
# ---------------------------------------------------------------------------


class Detection(BaseModel):
    """A single bounding-box detection from an object-detection model."""
    bbox: List[float]          # [x1, y1, x2, y2] in pixels
    score: float
    detector: str


class EmbeddingResult(BaseModel):
    """An embedding vector from a face / object recognition model."""
    vector: List[float]        # typically 512 floats (ArcFace)
    norm: float                # L2 norm before normalisation
    embedder: str


class RegionResult(BaseModel):
    """Results from sub-models run on a single detected region (crop).

    ``detection_index`` links back to the parent detection array.
    ``model_outputs`` holds the dynamic sub-model outputs keyed by category
    (e.g. ``{"face_embeddings": [{"vector": [...], ...}]}``).
    """
    detection_index: int
    model_outputs: Dict[str, Any] = {}


class ParsedImageData(BaseModel):
    """Normalised per-image result with tags, detections and regions separated."""
    tags: Dict[str, List] = {}                     # category -> [str] or [(str, float)]
    detections: Dict[str, List[Detection]] = {}    # category -> detections
    regions: Dict[str, List[RegionResult]] = {}    # regions__<alias> -> results
    error: Optional[str] = None


class ParsedFrameData(BaseModel):
    """Normalised per-frame structured data from a video response."""
    frame_index: float
    detections: Dict[str, List[Detection]] = {}
    regions: Dict[str, List[RegionResult]] = {}


class ParsedVideoData(BaseModel):
    """Fully parsed video response with tag timespans and structured frame data."""
    schema_version: int
    duration: float
    frame_interval: float
    models: List[AIModelInfo]
    timespans: Dict[str, Dict[str, List[TagTimeFrame]]]
    frames: Optional[List[ParsedFrameData]] = None


# ---------------------------------------------------------------------------
# Face processing intermediate types — used by face_processor.py
# ---------------------------------------------------------------------------


class FrameEmbedding(BaseModel):
    """A single face embedding captured at a specific frame timestamp."""
    vector: List[float]       # 512-dim, raw (not yet L2-normalised)
    norm: float               # L2 norm before normalisation
    score: float              # detection confidence at this frame
    bbox: List[float]         # [x1, y1, x2, y2] normalised
    timestamp_s: Optional[float] = None  # NULL for images
    embedder: str
    frame_index: Optional[float] = None  # frame sequence index (for sorting)


class TrackCandidate(BaseModel):
    """An assembled detection track from the track builder.

    For images, represents a single detection.
    For video, represents a temporally linked chain of detections.
    """
    label: str                          # 'face', 'person', etc.s
    best_bbox: List[float]              # best-quality bbox (normalised)
    best_score: float                   # highest detection confidence
    detector: str
    start_s: Optional[float] = None     # NULL for images
    end_s: Optional[float] = None       # NULL for images
    keyframes: Optional[List[Dict[str, Any]]] = None  # [{t, bbox}]
    embeddings: List[FrameEmbedding] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
