# Predictive Event Boundary Segmentation (PEBS) Event Slicer

Reusable video slicing tool for **event-based segmentation**. The project now contains only the core PEBS pipeline:

```text
video -> frame features -> predictive residuals -> event boundaries -> FFmpeg clips
```

No voice UI, no LLM parser, no command editing layer.

## What PEBS Does

PEBS means **Predictive Event Boundary Segmentation**. It treats a boundary as a prediction failure rather than raw frame difference.

For each sampled frame:

```text
x_t = [H_12, L_16, E_4, SV_2]
```

- `H_12`: 12-bin hue histogram
- `L_16`: 4x4 spatial brightness layout
- `E_4`: 2x2 edge density grid
- `SV_2`: global saturation and value

The predictor keeps a running visual context and recent motion:

```text
p_t = mu_(t-1) + delta_(t-1)
r_t = || x_t - p_t ||_2
```

Large residual spikes become boundary candidates. PEBS then:

- rescues strong fine-scale hard cuts
- refines hard-cut peaks back to the onset frame
- merges perceptually continuous adjacent pieces
- applies a minimum-duration prior
- exports clips with FFmpeg

## Install

From this folder:

```powershell
py -m pip install -e .
```

If `py` is not available, use your Python executable:

```powershell
python -m pip install -e .
```

The package uses `imageio-ffmpeg` as a fallback FFmpeg provider. You can also install FFmpeg yourself and put it on `PATH`.

## CLI Usage

```powershell
event-slice "path\to\video.mp4" -o "event_clips" --print-json
```

Or without installing:

```powershell
python -m event_slicer.cli "path\to\video.mp4" -o "event_clips"
```

Useful parameters:

```powershell
event-slice input.mp4 `
  --output-dir event_clips `
  --sample-fps 10 `
  --z-threshold 1.3 `
  --min-segment-seconds 0.8
```

Outputs:

```text
event_clips/
  input_part_01.mp4
  input_part_02.mp4
  ...
  index.json
```

## Python API

```python
from event_slicer import PEBSConfig, slice_video

result = slice_video(
    "input.mp4",
    "event_clips",
    pebs_config=PEBSConfig(
        sample_fps=10,
        z_threshold=1.3,
        min_segment_seconds=0.8,
    ),
)

print(result.boundary_times)
for segment in result.segments:
    print(segment.start, segment.end, segment.path)
```

Boundary detection only:

```python
from event_slicer import pebs_boundaries

result = pebs_boundaries("input.mp4")
print(result.boundary_times)
print(result.debug)
```

## Project Layout

```text
event_slicer/
  features.py   # frame -> compact feature vector
  pebs.py       # predictive residual boundary detection
  ffmpeg.py     # FFmpeg discovery, duration probe, segment export
  slicer.py     # public slice_video pipeline
  cli.py        # event-slice command
tests/
  test_pebs.py
  test_slicer.py
```

## Notes

- This is a lightweight event slicer, not a semantic captioning model.
- FFmpeg does all video writing; PEBS only decides boundary times.
- Default export re-encodes to H.264/AAC for reliable playback.
- `--stream-copy` is faster but can be less accurate around non-keyframe cuts.
