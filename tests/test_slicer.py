import tempfile
import unittest
from pathlib import Path

from event_slicer.models import Segment, SemanticSegmentResult, SliceResult
from event_slicer.slicer import safe_stem, segments_from_boundaries, write_index


class SlicerTests(unittest.TestCase):
    def test_safe_stem_removes_unsafe_characters(self):
        self.assertEqual(safe_stem(Path("my video (final!).mp4")), "my_video_final")

    def test_segments_from_boundaries_filters_tiny_segments(self):
        segments = segments_from_boundaries(10.0, [0.2, 2.0, 9.8], min_duration=1.0)
        self.assertEqual([(item.start, item.end) for item in segments], [(0.0, 2.0), (2.0, 10.0)])

    def test_write_index_serializes_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            result = SliceResult(
                input_path=Path("input.mp4"),
                output_dir=Path(tmp),
                duration=3.0,
                segmentation=SemanticSegmentResult([1.0], "pebs"),
                segments=[Segment(1, 0.0, 1.0, Path("part.mp4"))],
            )
            write_index(result, path)
            text = path.read_text(encoding="utf-8")
            self.assertIn('"boundary_times"', text)
            self.assertIn('"part.mp4"', text)


if __name__ == "__main__":
    unittest.main()
