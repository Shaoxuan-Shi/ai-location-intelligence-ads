from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_location import percentile_against_training, validate_candidate_coordinates
from utils import haversine_m, html_safe_json, median, point_in_polygon, quantile


class StatisticsTests(unittest.TestCase):
    def test_median_for_odd_and_even_sequences(self) -> None:
        self.assertEqual(median([3, 1, 2]), 2)
        self.assertEqual(median([1, 4, 2, 3]), 2.5)

    def test_quantile_interpolates(self) -> None:
        self.assertEqual(quantile([0, 10], 0.25), 2.5)

    def test_percentile_handles_ties_and_direction(self) -> None:
        values = [1, 2, 2, 4]
        self.assertEqual(percentile_against_training(2, values), 50.0)
        self.assertEqual(percentile_against_training(2, values, higher_is_better=False), 50.0)


class GeometryTests(unittest.TestCase):
    def test_haversine_is_zero_for_same_point(self) -> None:
        self.assertEqual(haversine_m(52.37, 4.89, 52.37, 4.89), 0.0)

    def test_haversine_amsterdam_reference_distance(self) -> None:
        distance = haversine_m(52.3731, 4.8922, 52.3791, 4.9003)
        self.assertTrue(math.isclose(distance, 865, rel_tol=0.08))

    def test_polygon_hole_is_excluded(self) -> None:
        polygon = [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],
        ]
        self.assertTrue(point_in_polygon(2, 2, polygon))
        self.assertFalse(point_in_polygon(5, 5, polygon))
        self.assertFalse(point_in_polygon(12, 2, polygon))


class ProductBoundaryTests(unittest.TestCase):
    def test_amsterdam_coordinate_is_accepted(self) -> None:
        validate_candidate_coordinates(52.3731, 4.8922)

    def test_out_of_scope_coordinate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "supported Amsterdam area"):
            validate_candidate_coordinates(51.5072, -0.1276)


class HtmlSerializationTests(unittest.TestCase):
    def test_script_closing_sequence_is_not_emitted(self) -> None:
        serialized = html_safe_json({"name": "</script><script>alert(1)</script>"})
        self.assertNotIn("</script", serialized.lower())
        self.assertIn("\\u003c/script\\u003e", serialized)

    def test_json_round_trip_preserves_text(self) -> None:
        import json

        payload = {"name": "A & B < C", "separator": "\u2028"}
        self.assertEqual(json.loads(html_safe_json(payload)), payload)


if __name__ == "__main__":
    unittest.main()
