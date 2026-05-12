from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PEBSConfig:
    """Configuration for predictive event-boundary segmentation."""

    sample_fps: float = 10.0
    z_threshold: float = 1.3
    min_segment_seconds: float = 0.8
    scale_seconds: float | None = None
    gestalt_similarity: float = 0.93


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for FFmpeg clip export."""

    ffmpeg_path: str | None = None
    reencode: bool = True
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    overwrite: bool = True


@dataclass
class SemanticSegmentResult:
    boundary_times: list[float]
    method: str
    fps: float = 0.0
    n_frames: int = 0
    debug: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    index: int
    start: float
    end: float
    path: Path | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "path": str(self.path) if self.path else None,
        }


@dataclass
class SliceResult:
    input_path: Path
    output_dir: Path
    duration: float
    segmentation: SemanticSegmentResult
    segments: list[Segment]

    @property
    def boundary_times(self) -> list[float]:
        return self.segmentation.boundary_times

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "duration": round(self.duration, 3),
            "segmentation": self.segmentation.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
        }
