"""Lazy-loaded CLIP text encoder for semantic visual search.

Loads only the text-projection half of MetaCLIP2 (ViT-H/14-quickgelu)
so we can encode natural-language queries into the same 1024-d space
as the stored visual embeddings, without loading any vision weights.

Requires ``transformers`` and ``torch`` — both are optional deps.
If unavailable the encoder returns ``None`` and the recommender
gracefully reports that text search is not available.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional

import numpy as np

_log = logging.getLogger(__name__)

_encoder: Optional[Callable] = None
_lock = threading.Lock()
_HF_MODEL = "facebook/metaclip-2-worldwide-huge-quickgelu"


def _load_encoder() -> Callable:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPTextModelWithProjection, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info("Loading CLIP text encoder %s on %s ...", _HF_MODEL, device)

    tokenizer = AutoTokenizer.from_pretrained(_HF_MODEL)
    model = CLIPTextModelWithProjection.from_pretrained(_HF_MODEL)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()

    def encode(texts: List[str]) -> np.ndarray:
        inp = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp)
            feats = out.text_embeds.float()
        return F.normalize(feats, dim=-1).cpu().numpy()

    _log.info("CLIP text encoder ready (dim=%d)", encode(["test"]).shape[1])
    return encode


def get_text_encoder() -> Optional[Callable]:
    """Return the cached text encoder, loading on first call."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _lock:
        if _encoder is not None:
            return _encoder
        try:
            _encoder = _load_encoder()
        except Exception:
            _log.exception("Failed to load CLIP text encoder — is transformers/torch installed?")
            return None
    return _encoder


def encode_text_query(query: str) -> Optional[List[float]]:
    """Encode a single text query into the MetaCLIP2 embedding space.

    Returns an L2-normalised 1024-d vector, or ``None`` on failure.
    """
    encoder = get_text_encoder()
    if encoder is None:
        return None
    vec = encoder([query])[0]  # shape: (D,)
    return vec.tolist()


def is_available() -> bool:
    """Quick check whether the required packages are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False
