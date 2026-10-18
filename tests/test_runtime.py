from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runtime_engine import evaluate_location, load_runtime_bundle
from agent_cli import run as run_agent_cli
from shortlist import build_shortlist


class RuntimeBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_runtime_bundle()

    def test_bundle_is_versioned_and_self_contained(self) -> None:
        runtime = self.bundle["runtime"]
        self.assertEqual(runtime["schema_version"], "1.0.0")
        self.assertEqual(runtime["geography"], "Amsterdam")
        self.assertRegex(runtime["data_as_of"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertGreater(len(self.bundle["pois"]), 1000)
        self.assertGreaterEqual(len(self.bundle["anchors"]), 20)

    def test_fresh_clone_runtime_evaluates_without_raw_data(self) -> None:
        result = evaluate_location(52.3731, 4.8922, "Test candidate", bundle=self.bundle)
        self.assertGreater(result["predicted_weekly_passers"], 0)
        self.assertLessEqual(result["prediction_low"], result["predicted_weekly_passers"])
        self.assertLessEqual(result["predicted_weekly_passers"], result["prediction_high"])
        self.assertEqual(result["runtime"]["data_as_of"], "2024-03-25")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for browser/Python parity validation")
    def test_python_runtime_matches_published_browser_runtime(self) -> None:
        html = (ROOT / "app" / "dashboard_agent.html").read_text(encoding="utf-8")
        context_match = re.search(
            r'<script id="dashboardContext" type="application/json">(.*?)</script>',
            html,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(context_match)
        browser_context = json.loads(context_match.group(1))
        start = html.index("    function log1p")
        end = html.index("    function bar", start)
        browser_runtime = html[start:end]

        for lat, lon in [(52.3731, 4.8922), (52.3791, 4.9003), (52.3602, 4.8851)]:
            script = (
                f"const ctx={json.dumps(browser_context, separators=(',', ':'))};"
                "const anchors=ctx.anchors;const pois=ctx.pois;const model=ctx.model;let current=null;\n"
                f"{browser_runtime}\n"
                f"const r=evaluate({lat},{lon},'Parity candidate');"
                "console.log(JSON.stringify({predicted:r.predicted,low:r.low,high:r.high,"
                "comparableError:r.comparableError,variability:r.variability,"
                "outside:r.outsideFeatureCount,support:r.supportCount,similarity:r.jointSimilarityPct}));"
            )
            completed = subprocess.run(
                ["node", "-"],
                input=script,
                text=True,
                capture_output=True,
                check=True,
            )
            browser = json.loads(completed.stdout)
            python = evaluate_location(lat, lon, "Parity candidate", bundle=self.bundle)
            self.assertAlmostEqual(browser["predicted"], python["predicted_weekly_passers"], places=5)
            self.assertAlmostEqual(browser["low"], python["prediction_low"], places=5)
            self.assertAlmostEqual(browser["high"], python["prediction_high"], places=5)
            self.assertAlmostEqual(browser["comparableError"], python["similar_anchor_validation_error"], places=9)
            self.assertAlmostEqual(browser["variability"], python["comparable_anchor_variability"], places=9)
            self.assertEqual(browser["outside"], python["outside_feature_count"])
            self.assertEqual(browser["support"], python["feature_support_count"])
            self.assertEqual(browser["similarity"], python["joint_feature_similarity_percentile"])


class ShortlistContractTests(unittest.TestCase):
    def candidate(self, name: str, predicted: float, context: float = 80, **overrides) -> dict:
        row = {
            "name": name,
            "predicted_weekly_passers": predicted,
            "tourism_percentile": context,
            "retail_percentile": context,
            "transit_percentile": context,
            "outside_feature_count": 0,
            "similar_anchor_validation_error": 0.4,
            "comparable_anchor_variability": 0.3,
            "joint_feature_similarity_percentile": 50,
            "feature_support_count": 2,
        }
        row.update(overrides)
        return row

    def test_filter_first_then_rank_by_predicted_exposure(self) -> None:
        high_but_wrong_context = self.candidate("High excluded", 500_000, context=20)
        eligible_high = self.candidate("Eligible high", 300_000)
        eligible_low = self.candidate("Eligible low", 100_000)
        result = build_shortlist(
            [eligible_low, high_but_wrong_context, eligible_high],
            {"objective": "tourism", "evidence_policy": "balanced", "shortlist_size": 2},
        )
        self.assertEqual([row["name"] for row in result["selected"]], ["Eligible high", "Eligible low"])
        self.assertEqual(result["excluded"][0]["candidate"]["name"], "High excluded")

    def test_strict_policy_exposes_every_failed_rule(self) -> None:
        candidate = self.candidate(
            "Weak evidence",
            200_000,
            outside_feature_count=2,
            similar_anchor_validation_error=0.9,
            comparable_anchor_variability=0.6,
            joint_feature_similarity_percentile=5,
            feature_support_count=0,
        )
        result = build_shortlist(
            [candidate],
            {"objective": "exposure", "evidence_policy": "strict", "shortlist_size": 3},
        )
        self.assertFalse(result["selected"])
        self.assertEqual(len(result["excluded"][0]["reasons"]), 5)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for shortlist parity validation")
    def test_python_shortlist_matches_published_browser_rules(self) -> None:
        candidates = [
            self.candidate("High excluded", 500_000, context=20),
            self.candidate("Eligible high", 300_000),
            self.candidate("Eligible low", 100_000, similar_anchor_validation_error=0.8),
        ]
        brief = {"objective": "tourism", "evidence_policy": "strict", "shortlist_size": 2}
        python_result = build_shortlist(candidates, brief)
        browser_candidates = [
            {
                "name": row["name"],
                "predicted": row["predicted_weekly_passers"],
                "tourismPct": row["tourism_percentile"],
                "retailPct": row["retail_percentile"],
                "transitPct": row["transit_percentile"],
                "outsideFeatureCount": row["outside_feature_count"],
                "comparableError": row["similar_anchor_validation_error"],
                "variability": row["comparable_anchor_variability"],
                "jointSimilarityPct": row["joint_feature_similarity_percentile"],
                "supportCount": row["feature_support_count"],
            }
            for row in candidates
        ]
        browser_brief = {
            "objective": brief["objective"],
            "evidencePolicy": brief["evidence_policy"],
            "shortlistSize": brief["shortlist_size"],
        }
        html = (ROOT / "app" / "dashboard_agent.html").read_text(encoding="utf-8")
        start = html.index("    function objectiveReasons")
        end = html.index("    function renderSuggestions", start)
        browser_rules = html[start:end]
        script = (
            f"const savedCandidates={json.dumps(browser_candidates)};\n"
            f"{browser_rules}\n"
            f"const r=buildShortlist({json.dumps(browser_brief)});"
            "console.log(JSON.stringify({selected:r.selected.map(x=>x.name),"
            "excluded:r.excluded.map(x=>({name:x.candidate.name,reasons:x.reasons}))}));"
        )
        completed = subprocess.run(
            ["node", "-"], input=script, text=True, capture_output=True, check=True
        )
        browser_result = json.loads(completed.stdout)
        self.assertEqual(browser_result["selected"], [row["name"] for row in python_result["selected"]])
        self.assertEqual(
            browser_result["excluded"],
            [
                {"name": item["candidate"]["name"], "reasons": item["reasons"]}
                for item in python_result["excluded"]
            ],
        )


class AgentBoundaryTests(unittest.TestCase):
    def test_unsupported_business_decision_returns_no_prediction(self) -> None:
        result = run_agent_cli("What are the CPM and ROI for Damrak 1-5?")
        self.assertEqual(result["kind"], "unsupported_decision")
        self.assertEqual(result["decision_status"], "not_buy_ready")
        self.assertIn("price_or_cpm", result["unsupported_fields"])
        self.assertIn("campaign_roi", result["unsupported_fields"])
        self.assertNotIn("candidate", result)
        self.assertNotIn("selected", result)

    def test_live_and_chinese_buying_requests_are_rejected(self) -> None:
        for query in ["What is the live foot traffic right now?", "这个点位现在可以直接购买吗？"]:
            with self.subTest(query=query):
                result = run_agent_cli(query)
                self.assertEqual(result["kind"], "unsupported_decision")
                self.assertFalse({"selected", "candidate"} & result.keys())

    def test_location_output_explains_interval_freshness_and_next_steps(self) -> None:
        result = run_agent_cli("Estimate 52.3731 4.8922", candidate_name="Dam test")
        candidate = result["candidate"]
        self.assertEqual(candidate["name"], "Dam test")
        self.assertEqual(candidate["interval_coverage"], "approximate_80_percent")
        self.assertIn("strict", candidate["evidence_policy_results"])
        self.assertEqual(candidate["data_freshness"]["status"], "historical_requires_current_check")
        self.assertGreaterEqual(len(candidate["field_validation_priorities"]), 4)

    def test_shortlist_declares_candidate_universe_and_screening_status(self) -> None:
        result = run_agent_cli("Build a strict tourism top 3 shortlist")
        self.assertEqual(result["decision_status"], "screening_only")
        self.assertFalse(result["candidate_universe"]["is_live_zap_inventory"])
        self.assertEqual(result["candidate_universe"]["type"], "historical_training_anchors")
        self.assertEqual(result["selected"], result["field_validation_candidates"])
