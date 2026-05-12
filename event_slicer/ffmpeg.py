from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from .models import ExportConfig, Segment


def find_ffmpeg(configured: str | None = None) -> str | None:
    if configured and Path(configured).exists():
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def require_ffmpeg(configured: str | None = None) -> str:
    executable = find_ffmpeg(configured)
    if not executable:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg or set FFMPEG_PATH.")
    return executable


def run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(output or f"FFmpeg failed with exit code {result.returncode}")


def probe_duration(path: Path, ffmpeg_path: str | None = None) -> float:
    executable = require_ffmpeg(ffmpeg_path)
    result = subprocess.run(
        [executable, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stderr}\n{result.stdout}"
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not match:
        raise RuntimeError(f"Could not read video duration: {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def export_segment(source: Path, segment: Segment, output_path: Path, config: ExportConfig | None = None) -> Path:
    config = config or ExportConfig()
    executable = require_ffmpeg(config.ffmpeg_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        executable,
        "-y" if config.overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ss",
        f"{segment.start:.3f}",
        "-t",
        f"{segment.duration:.3f}",
    ]
    if config.reencode:
        command += [
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            config.video_codec,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            config.audio_codec,
            "-movflags",
            "+faststart",
        ]
    else:
        command += ["-map", "0", "-c", "copy"]
    command.append(str(output_path))
    run_ffmpeg(command)
    segment.path = output_path
    return output_path
