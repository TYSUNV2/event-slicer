from __future__ import annotations

import numpy as np

from .features import read_video_features
from .models import PEBSConfig, SemanticSegmentResult


def predictive_surprise(features: np.ndarray, alpha: float = 0.18, beta: float = 0.08) -> np.ndarray:
    """Return the prediction residual for every sampled frame feature."""

    count, dims = features.shape
    mu = features[0].copy()
    delta = np.zeros(dims, dtype=np.float32)
    surprise = np.zeros(count, dtype=np.float32)

    for index in range(1, count):
        prediction = mu + delta
        surprise[index] = float(np.linalg.norm(features[index] - prediction))
        new_delta = features[index] - mu
        delta = beta * new_delta + (1.0 - beta) * delta
        mu = alpha * features[index] + (1.0 - alpha) * mu

    return surprise


def smooth(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal, kernel, mode="same")


def robust_z_scores(signal: np.ndarray) -> np.ndarray:
    median = float(np.median(signal))
    mad = float(np.median(np.abs(signal - median))) + 1e-6
    return (signal - median) / (1.4826 * mad)


def find_peaks_nms(signal: np.ndarray, min_dist: int, z_threshold: float = 1.3) -> list[int]:
    if len(signal) < 3:
        return []

    z_scores = robust_z_scores(signal)
    candidates = [
        (index, float(z_scores[index]))
        for index in range(1, len(signal) - 1)
        if z_scores[index] > z_threshold
        and signal[index] > signal[index - 1]
        and signal[index] >= signal[index + 1]
    ]
    candidates.sort(key=lambda item: -item[1])

    selected: list[int] = []
    for index, _ in candidates:
        if all(abs(index - existing) >= min_dist for existing in selected):
            selected.append(index)
    return sorted(selected)


def boundary_contrast(features: np.ndarray, boundary: int, fps: float, window_sec: float = 1.0) -> dict[str, float]:
    window = max(int(window_sec * fps), 3)
    left_start = max(0, boundary - window)
    right_end = min(len(features), boundary + window)
    if boundary <= left_start or right_end <= boundary:
        return {
            "color_cos": 1.0,
            "layout_max_l1": 0.0,
            "edge_max_l1": 0.0,
            "saturation_diff": 0.0,
            "value_diff": 0.0,
            "feature_l2": 0.0,
        }

    left = features[left_start:boundary].mean(axis=0)
    right = features[boundary:right_end].mean(axis=0)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    return {
        "color_cos": cosine(left[0:12], right[0:12]),
        "layout_max_l1": float(np.max(np.abs(left[12:28] - right[12:28]))),
        "edge_max_l1": float(np.max(np.abs(left[28:32] - right[28:32]))),
        "saturation_diff": float(abs(left[32] - right[32])),
        "value_diff": float(abs(left[33] - right[33])),
        "feature_l2": float(np.linalg.norm(left - right)),
    }


def refine_boundary_onset(
    features: np.ndarray,
    boundary: int,
    fps: float,
    search_back_sec: float = 0.7,
    search_forward_sec: float = 0.2,
) -> int:
    """Move a smoothed surprise peak back to the first sharp visual jump."""

    if len(features) < 3:
        return boundary

    delta = np.zeros(len(features), dtype=np.float32)
    delta[1:] = np.linalg.norm(np.diff(features, axis=0), axis=1)
    start = max(1, boundary - int(search_back_sec * fps))
    end = min(len(features) - 1, boundary + int(search_forward_sec * fps))
    if end <= start:
        return boundary
    local = delta[start:end + 1]
    return int(start + int(np.argmax(local)))


def rescue_hard_boundaries(
    surprise: np.ndarray,
    features: np.ndarray,
    fps: float,
    scale_sec: float,
    z_threshold: float,
) -> tuple[list[int], list[dict[str, float]]]:
    """Keep strong fine-scale surprise events that coarse smoothing can erase."""

    rescue_scale = min(max(scale_sec / 3.0, 0.25), 0.6)
    window = max(int(rescue_scale * fps), 3)
    signal = smooth(surprise, window)
    min_dist = max(int(2.0 * fps), 3)
    rescue_z = max(z_threshold + 0.8, 2.1)
    peaks = find_peaks_nms(signal, min_dist=min_dist, z_threshold=rescue_z)
    z_scores = robust_z_scores(signal)

    selected: list[int] = []
    debug: list[dict[str, float]] = []
    for peak in peaks:
        contrast = boundary_contrast(features, peak, fps)
        hard_visual_jump = (
            contrast["feature_l2"] >= 0.38
            and (
                contrast["color_cos"] <= 0.90
                or contrast["layout_max_l1"] >= 0.16
                or contrast["edge_max_l1"] >= 0.09
                or contrast["saturation_diff"] >= 0.12
                or contrast["value_diff"] >= 0.12
            )
        )
        if float(z_scores[peak]) >= rescue_z and hard_visual_jump:
            refined = refine_boundary_onset(features, peak, fps)
            if any(abs(refined - existing) < max(int(0.4 * fps), 2) for existing in selected):
                continue
            selected.append(refined)
            debug.append(
                {
                    "time": round(refined / fps, 3),
                    "peak_time": round(peak / fps, 3),
                    "z": round(float(z_scores[peak]), 3),
                    "scale_sec": round(rescue_scale, 3),
                    **{key: round(value, 4) for key, value in contrast.items()},
                }
            )

    return selected, debug


def hierarchical_boundaries(
    surprise: np.ndarray,
    fps: float,
    scales_sec: tuple[float, ...],
    z_threshold: float,
) -> dict[float, list[int]]:
    output: dict[float, list[int]] = {}
    for scale in scales_sec:
        window = max(int(scale * fps), 3)
        min_dist = max(int(0.6 * scale * fps), 3)
        output[scale] = find_peaks_nms(smooth(surprise, window), min_dist, z_threshold)
    return output


def gestalt_merge(
    boundaries: list[int],
    features: np.ndarray,
    similarity: float = 0.93,
    region_max_l1: float = 0.07,
) -> list[int]:
    if not boundaries:
        return []

    cuts = [0] + sorted(boundaries) + [len(features)]
    centroids = [features[cuts[index]:cuts[index + 1]].mean(axis=0) for index in range(len(cuts) - 1)]

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))

    kept: list[int] = []
    for index, boundary in enumerate(boundaries):
        left = centroids[index]
        right = centroids[index + 1]
        perceptually_same = (
            cosine(left[0:12], right[0:12]) > similarity
            and float(np.max(np.abs(left[12:28] - right[12:28]))) < region_max_l1
            and float(np.max(np.abs(left[28:32] - right[28:32]))) < region_max_l1
        )
        if not perceptually_same:
            kept.append(boundary)
    return kept


def apply_duration_prior(boundaries: list[int], total: int, fps: float, d_min: float = 0.8) -> list[int]:
    if not boundaries:
        return []
    cuts = [0] + sorted(boundaries) + [total]
    kept = [cuts[0]]
    for cut in cuts[1:]:
        if (cut - kept[-1]) / fps >= d_min:
            kept.append(cut)
    if kept[-1] != total:
        if len(kept) >= 2:
            kept[-1] = total
        else:
            kept.append(total)
    return kept[1:-1]


def detect_boundaries(features: np.ndarray, fps: float, config: PEBSConfig | None = None) -> SemanticSegmentResult:
    config = config or PEBSConfig()
    if len(features) < 6:
        return SemanticSegmentResult([], "pebs-too-short", fps, len(features))

    duration = len(features) / fps
    scale_sec = config.scale_seconds
    if scale_sec is None:
        scale_sec = float(np.clip(duration / 20.0, 0.4, 4.0))

    surprise = predictive_surprise(features)
    fine = max(0.25, scale_sec / 3)
    coarse = min(4.0, max(scale_sec * 2.5, scale_sec + 0.5))
    if fine >= scale_sec:
        fine = scale_sec / 2
    scales = tuple(dict.fromkeys((fine, scale_sec, coarse)))
    hierarchy = hierarchical_boundaries(surprise, fps, scales, config.z_threshold)

    working = list(hierarchy.get(scale_sec, []))
    if not working and hierarchy:
        working = list(next(iter(hierarchy.values())))

    rescued, rescued_debug = rescue_hard_boundaries(surprise, features, fps, scale_sec, config.z_threshold)
    working = sorted(set(working + rescued))
    working = gestalt_merge(working, features, similarity=config.gestalt_similarity)
    working = apply_duration_prior(working, len(features), fps, d_min=config.min_segment_seconds)

    return SemanticSegmentResult(
        boundary_times=[round(boundary / fps, 3) for boundary in working],
        method="pebs",
        fps=fps,
        n_frames=len(features),
        debug={
            "scale_sec": scale_sec,
            "scales": list(scales),
            "hierarchy": {
                str(scale): [round(boundary / fps, 3) for boundary in boundaries]
                for scale, boundaries in hierarchy.items()
            },
            "hard_rescue": rescued_debug,
        },
    )


def pebs_boundaries(video_path: str, config: PEBSConfig | None = None, **overrides: float) -> SemanticSegmentResult:
    """Detect PEBS event boundaries in seconds for a video path.

    Keyword overrides keep the public API convenient for notebooks:
    `pebs_boundaries("video.mp4", z_threshold=1.5, sample_fps=8)`.
    """

    if config is None:
        values = PEBSConfig()
        config = PEBSConfig(
            sample_fps=float(overrides.get("sample_fps", values.sample_fps)),
            z_threshold=float(overrides.get("z_threshold", values.z_threshold)),
            min_segment_seconds=float(overrides.get("min_segment_seconds", overrides.get("d_min", values.min_segment_seconds))),
            scale_seconds=overrides.get("scale_seconds", overrides.get("scale_sec", values.scale_seconds)),
            gestalt_similarity=float(overrides.get("gestalt_similarity", values.gestalt_similarity)),
        )
    features, feature_fps = read_video_features(video_path, sample_fps=config.sample_fps)
    return detect_boundaries(features, feature_fps, config)
