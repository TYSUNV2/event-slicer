from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import ExportConfig, PEBSConfig
from .slicer import slice_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="event-slice",
        description="Slice a video into PEBS event clips.",
    )
    parser.add_argument("input", help="Input video path")
    parser.add_argument("-o", "--output-dir", default="event_clips", help="Directory for exported clips")
    parser.add_argument("--sample-fps", type=float, default=10.0, help="Frame sampling rate used by PEBS")
    parser.add_argument("--z-threshold", type=float, default=1.3, help="Robust z-score threshold for residual peaks")
    parser.add_argument("--min-segment-seconds", type=float, default=0.8, help="Minimum clip duration")
    parser.add_argument("--scale-seconds", type=float, default=None, help="Override PEBS event scale")
    parser.add_argument("--ffmpeg-path", default=None, help="Optional path to ffmpeg executable")
    parser.add_argument("--stream-copy", action="store_true", help="Use -c copy instead of re-encoding")
    parser.add_argument("--no-json", action="store_true", help="Do not write index.json")
    parser.add_argument("--print-json", action="store_true", help="Print the result JSON to stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pebs_config = PEBSConfig(
        sample_fps=args.sample_fps,
        z_threshold=args.z_threshold,
        min_segment_seconds=args.min_segment_seconds,
        scale_seconds=args.scale_seconds,
    )
    export_config = ExportConfig(
        ffmpeg_path=args.ffmpeg_path,
        reencode=not args.stream_copy,
    )
    result = slice_video(
        Path(args.input),
        Path(args.output_dir),
        pebs_config=pebs_config,
        export_config=export_config,
        write_json=not args.no_json,
    )
    if args.print_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Exported {len(result.segments)} clip(s) to {result.output_dir}")
        print(f"Boundaries: {result.boundary_times}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
