from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_dashboard_and_model_artifact_exist(self) -> None:
        self.assertTrue((ROOT / "app" / "dashboard_agent.html").is_file())
        self.assertTrue((ROOT / "outputs" / "tables" / "model_artifact.json").is_file())

    def test_model_artifact_has_required_fields(self) -> None:
        artifact = json.loads((ROOT / "outputs" / "tables" / "model_artifact.json").read_text())
        required = {"coefficients", "features", "training_rows", "interval_log_margin"}
        self.assertTrue(required.issubset(artifact))
        self.assertEqual(len(artifact["coefficients"]), len(artifact["features"]) + 1)
        self.assertGreaterEqual(len(artifact["training_rows"]), 20)

    def test_prediction_table_schema_and_bounds(self) -> None:
        with (ROOT / "outputs" / "tables" / "site_predictions_weekly.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 20)
        required = {
            "location_name",
            "lat",
            "lon",
            "predicted_weekly_passers",
            "prediction_low",
            "prediction_high",
            "loocv_absolute_percentage_error",
        }
        self.assertTrue(required.issubset(rows[0]))
        for row in rows:
            low = float(row["prediction_low"])
            point = float(row["predicted_weekly_passers"])
            high = float(row["prediction_high"])
            self.assertLessEqual(low, point)
            self.assertLessEqual(point, high)

    def test_published_dashboard_contains_product_boundaries(self) -> None:
        html = (ROOT / "app" / "dashboard_agent.html").read_text(encoding="utf-8")
        for phrase in [
            "Reliability Evidence",
            "Campaign Shortlist Builder",
            "field validation",
            "I cannot access inventory price",
        ]:
            self.assertIn(phrase, html)

    def test_candidate_name_is_escaped_in_html_sinks(self) -> None:
        files = [
            ROOT / "src" / "generate_dashboard.py",
            ROOT / "src" / "generate_dashboard_assistant_v1.py",
            ROOT / "app" / "dashboard_agent.html",
            ROOT / "app" / "dashboard.html",
        ]
        unsafe_patterns = [
            "<h3>${result.name}</h3>",
            "bindPopup(`<strong>${result.name}</strong>",
        ]
        for path in files:
            text = path.read_text(encoding="utf-8")
            for pattern in unsafe_patterns:
                self.assertNotIn(pattern, text, str(path))
            self.assertIn("<h3>${escapeHtml(result.name)}</h3>", text, str(path))

    def test_anchor_names_are_escaped_in_html_sinks(self) -> None:
        files = [
            ROOT / "src" / "generate_dashboard.py",
            ROOT / "src" / "generate_dashboard_assistant_v1.py",
            ROOT / "app" / "dashboard_agent.html",
            ROOT / "app" / "dashboard.html",
        ]
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn('<tr data-name="${anchor.name}">', text, str(path))
            self.assertNotIn('<td>${anchor.name}</td>', text, str(path))
            self.assertNotIn('<strong>${anchor.name}</strong>', text, str(path))
            self.assertIn("${escapeHtml(anchor.name)}", text, str(path))

    def test_generated_dashboards_use_non_executable_json_context(self) -> None:
        for relative in ["app/dashboard_agent.html", "app/dashboard.html"]:
            html = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn('id="dashboardContext" type="application/json"', html)
            self.assertIn("JSON.parse(document.getElementById('dashboardContext').textContent)", html)

    def test_external_executable_resources_have_sri(self) -> None:
        for relative in ["app/dashboard_agent.html", "app/dashboard.html"]:
            html = (ROOT / relative).read_text(encoding="utf-8")
            self.assertEqual(html.count("https://unpkg.com/"), 5)
            self.assertEqual(html.count('integrity="sha384-'), 5)
            self.assertEqual(html.count('crossorigin="anonymous"'), 5)

    def test_browser_enforces_amsterdam_coordinate_boundary(self) -> None:
        for relative in ["app/dashboard_agent.html", "app/dashboard.html"]:
            html = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("function isSupportedCoordinate(lat, lon)", html)
            self.assertIn("lat >= 52.30 && lat <= 52.43", html)

    def test_fetch_pipeline_has_no_workspace_sibling_dependency(self) -> None:
        source = (ROOT / "src" / "fetch_data.py").read_text(encoding="utf-8")
        self.assertNotIn("ROOT.parent", source)
        self.assertNotIn("Intern_Assignment", source)


if __name__ == "__main__":
    unittest.main()
