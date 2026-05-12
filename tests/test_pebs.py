import unittest

import numpy as np

from event_slicer.models import PEBSConfig
from event_slicer.pebs import apply_duration_prior, detect_boundaries, predictive_surprise, refine_boundary_onset


class PEBSTests(unittest.TestCase):
    def test_predictive_surprise_spikes_on_unexpected_feature(self):
        features = np.zeros((12, 4), dtype=np.float32)
        features[6:] = 1.0
        surprise = predictive_surprise(features)
        self.assertGreater(float(surprise[6]), float(surprise[5]))

    def test_duration_prior_removes_tiny_first_segment(self):
        boundaries = apply_duration_prior([6, 18], total=66, fps=10.0, d_min=0.8)
        self.assertEqual(boundaries, [18])

    def test_refine_boundary_moves_smoothed_peak_to_cut_onset(self):
        features = np.zeros((30, 4), dtype=np.float32)
        features[12:] = 1.0
        self.assertEqual(refine_boundary_onset(features, boundary=15, fps=10.0), 12)

    def test_detect_boundaries_returns_result_object(self):
        features = np.zeros((80, 34), dtype=np.float32)
        features[30:] = 1.0
        result = detect_boundaries(features, fps=10.0, config=PEBSConfig(z_threshold=1.0))
        self.assertEqual(result.method, "pebs")
        self.assertIsInstance(result.boundary_times, list)
        self.assertEqual(result.n_frames, 80)


if __name__ == "__main__":
    unittest.main()
