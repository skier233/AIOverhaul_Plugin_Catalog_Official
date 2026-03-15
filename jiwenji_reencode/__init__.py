from __future__ import annotations

# ---------------------------------------------------------------------------
# Codec family map — used by both encoder (worker) and service (backend)
# to match ffprobe/Stash codec names against user-selected skip families.
# ---------------------------------------------------------------------------
CODEC_FAMILIES: dict[str, frozenset[str]] = {
    "hevc": frozenset({"hevc", "h265", "h.265"}),
    "av1":  frozenset({"av1"}),
    "vp9":  frozenset({"vp9"}),
    "vp8":  frozenset({"vp8"}),
}
