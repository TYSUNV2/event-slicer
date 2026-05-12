"""Reusable event-based video slicing toolkit.

PEBS is a lightweight predictive event-boundary segmenter. It detects boundaries
from prediction residuals over compact frame features, then uses FFmpeg to
export clips.
"""

from .models import ExportConfig, PEBSConfig, Segment, SemanticSegmentResult, SliceResult
from .pebs import detect_boundaries, pebs_boundaries
from .slicer import slice_video

__all__ = [
    "ExportConfig",
    "PEBSConfig",
    "Segment",
    "SemanticSegmentResult",
    "SliceResult",
    "detect_boundaries",
    "pebs_boundaries",
    "slice_video",
]
