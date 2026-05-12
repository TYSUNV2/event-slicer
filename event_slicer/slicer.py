from __future__ import annotations

import json
import re
from pathlib import Path

from .ffmpeg import export_segment, probe_duration
from .models import ExportConfig, PEBSConfig, Segment, SliceResult
from .pebs import pebs_boundaries

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "video"


def segments_from_boundaries(duration: float, boundaries: list[float], min_duration: float) -> list[Segment]:
    cuts = [0.0]
    for point in sorted(float(item) for item in boundaries):
        if min_duration <= point <= duration - min_duration and point - cuts[-1] >= min_duration:
            cuts.append(round(point, 3))
    if duration - cuts[-1] < min_duration and len(cuts) > 1:
        cuts.pop()
    cuts.append(round(duration, 3))
    return [
        Segment(index=index, start=start, end=end)
        for index, (start, end) in enumerate(zip(cuts, cuts[1:]), start=1)
        if end - start > 0.1
    ]


def write_index(result: SliceResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def slice_video(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    pebs_config: PEBSConfig | None = None,
    export_config: ExportConfig | None = None,
    write_json: bool = True,
) -> SliceResult:
    """Detect event boundaries with PEBS and export clips with FFmpeg."""

    source = Path(input_path).resolve()
    if source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {source.suffix}")
    if not source.exists():
        raise FileNotFoundError(source)

    pebs_config = pebs_config or PEBSConfig()
    export_config = export_config or ExportConfig()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(source, export_config.ffmpeg_path)
    segmentation = pebs_boundaries(str(source), pebs_config)
    segments = segments_from_boundaries(duration, segmentation.boundary_times, pebs_config.min_segment_seconds)

    stem = safe_stem(source)
    for segment in segments:
        clip_path = output / f"{stem}_part_{segment.index:02d}.mp4"
        export_segment(source, segment, clip_path, export_config)

    result = SliceResult(
        input_path=source,
        output_dir=output,
        duration=duration,
        segmentation=segmentation,
        segments=segments,
    )
    if write_json:
        write_index(result, output / "index.json")
    return result
