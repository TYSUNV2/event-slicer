from __future__ import annotations

import numpy as np


def extract_features(frame: np.ndarray) -> np.ndarray:
    """Encode one BGR frame as a compact 34-D PEBS feature vector."""

    import cv2

    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    hue_hist = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    hue_hist /= hue_hist.sum() + 1e-9

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    layout = np.zeros(16, dtype=np.float32)
    for row in range(4):
        for col in range(4):
            cell = gray[
                row * height // 4:(row + 1) * height // 4,
                col * width // 4:(col + 1) * width // 4,
            ]
            layout[row * 4 + col] = cell.mean() / 255.0

    edges = cv2.Canny(gray, 50, 150)
    edge_grid = np.zeros(4, dtype=np.float32)
    for row in range(2):
        for col in range(2):
            cell = edges[
                row * height // 2:(row + 1) * height // 2,
                col * width // 2:(col + 1) * width // 2,
            ]
            edge_grid[row * 2 + col] = cell.mean() / 255.0

    sat_mean = float(hsv[..., 1].mean()) / 255.0
    val_mean = float(hsv[..., 2].mean()) / 255.0

    return np.concatenate(
        [hue_hist, layout, edge_grid, np.array([sat_mean, val_mean], dtype=np.float32)]
    ).astype(np.float32)


def read_video_features(video_path: str, sample_fps: float = 10.0) -> tuple[np.ndarray, float]:
    """Sample a video and return PEBS features plus the actual feature FPS."""

    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(int(round(fps / sample_fps)), 1)
    features = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step == 0:
            features.append(extract_features(frame))
        frame_index += 1
    cap.release()

    feature_fps = fps / step
    if not features:
        return np.zeros((0, 34), dtype=np.float32), feature_fps
    return np.stack(features).astype(np.float32), feature_fps
