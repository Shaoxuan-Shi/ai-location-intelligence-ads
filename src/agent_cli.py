from __future__ import annotations

import argparse
from datetime import date
import json
import re
from typing import Any

from runtime_engine import evaluate_location, load_runtime_bundle
from shortlist import build_shortlist, evidence_reasons, normalize_brief, parse_campaign_brief


UNSUPPORTED_PATTERNS = {
    "media_buying_decision": r"\b(?:buy|purchase|approve|book|proceed)\b|直接购买|购买|批准|下单|投放",
    "price_or_cpm": r"\b(?:cpm|price|pricing|cost)\b|价格|成本|千次展示",
    "campaign_roi": r"\b(?:roi|return on investment|conversion|campaign lift)\b|投资回报|回报率|转化|广告效果",
    "inventory_availability": r"\b(?:inventory|availability|available screen|vacancy)\b|库存|可售|档期|空位",
    "live_or_hourly_traffic": r"\b(?:live|right now|real[ -]?time|hourly traffic|traffic now)\b|实时|现在人流|当前人流|小时级|每小时",
    "audience_demographics": r"\b(?:demographic|audience profile|age group)\b|人口属性|人群画像|受众画像",
    "screen_visibility": r"\b(?:screen visibility|orientation|sight ?line|obstruction)\b|屏幕可见|屏幕朝向|视线|遮挡",
}

MISSING_INPUTS = {
    "media_buying_decision": ["price and contract terms", "inventory availability", "screen-level field validation"],
    "price_or_cpm": ["media price", "booked impressions or a defensible exposure denominator"],
    "campaign_roi": ["campaign cost", "conversion outcome", "causal measurement design"],
    "inventory_availability": ["current ZAP inventory and booking calendar"],
    "live_or_hourly_traffic": ["current or hourly pedestrian-count feed"],
    "audience_demographics": ["privacy-safe audience or survey data"],
    "screen_visibility": ["screen orientation, sight line, obstruction, and viewing-distance observations"],
}


def unsupported_fields(query: str) -> list[str]:
    return [name for name, pattern in UNSUPPORTED_PATTERNS.items() if re.search(pattern, query.lower())]


def unsupported_response(query: str, bundle: dict[str, Any]) -> dict[str, Any] | None:
    fields = unsupported_fields(query)
    if not fields:
        return None
    missing = []
    for field in fields:
        for item in MISSING_INPUTS[field]:
            if item not in missing:
                missing.append(item)
    return {
        "kind": "unsupported_decision",
        "decision_status": "not_buy_ready",
        "boundary": "The current model cannot answer this request and no estimate or shortlist was generated.",
        "unsupported_fields": fields,
        "missing_inputs": missing,
        "allowed_next_steps": [
            "Use the model only to prioritize candidates for field validation.",
            "Collect the missing operational inputs before making a media-buying decision.",
            "Use a current pedestrian-count source for live or hourly traffic questions.",
        ],
        "runtime": bundle.get("runtime", {}),
    }


def extract_lat_lon(query: str) -> tuple[float, float] | None:
    numbers = [float(item) for item in re.findall(r"-?\d+\.\d+", query)]
    for index in range(len(numbers) - 1):
        lat, lon = numbers[index], numbers[index + 1]
        if 52.0 <= lat <= 52.6 and 4.5 <= lon <= 5.3:
            return lat, lon
    return None


def find_anchor(query: str, anchors: list[dict[str, Any]]) -> dict[str, Any] | None:
    q = query.lower()
    exact = [anchor for anchor in anchors if str(anchor["name"]).lower() in q]
    if exact:
        return max(exact, key=lambda row: len(str(row["name"])))
    tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", q) if len(token) >= 4]
    for anchor in anchors:
        name = str(anchor["name"]).lower()
        if any(token in name for token in tokens):
            return anchor
    return None


def public_candidate(row: dict[str, Any]) -> dict[str, Any]:
    runtime = row.get("runtime", {})
    data_as_of = runtime.get("data_as_of")
    data_age_days = None
    if data_as_of:
        try:
            data_age_days = (date.today() - date.fromisoformat(data_as_of)).days
        except ValueError:
            data_age_days = None
    policy_results = {}
    for policy in ["strict", "balanced", "exploratory"]:
        reasons = evidence_reasons(row, normalize_brief({"evidence_policy": policy}))
        policy_results[policy] = {"passes": not reasons, "reasons": reasons}
    strict_passes = policy_results["strict"]["passes"]
    return {
        "name": row["name"],
        "lat": row["lat"],
        "lon": row["lon"],
        "predicted_weekly_passers": round(row["predicted_weekly_passers"]),
        "prediction_low": round(row["prediction_low"]),
        "prediction_high": round(row["prediction_high"]),
        "interval_coverage": row["interval_coverage"],
        "interval_method": row["interval_method"],
        "interval_width_ratio": round(row["interval_width_ratio"], 3),
        "retail_percentile": row["retail_percentile"],
        "horeca_percentile": row["horeca_percentile"],
        "transit_percentile": row["transit_percentile"],
        "tourism_percentile": row["tourism_percentile"],
        "similar_anchor_validation_error_pct": round(row["similar_anchor_validation_error"] * 100, 1),
        "historical_spread_pct": round(row["comparable_anchor_variability"] * 100, 1),
        "outside_feature_count": row["outside_feature_count"],
        "feature_support_count": row["feature_support_count"],
        "joint_feature_similarity_percentile": row["joint_feature_similarity_percentile"],
        "shopping_area_name": row["shopping_area_name"] or None,
        "evidence_policy_results": policy_results,
        "evidence_interpretation": (
            "Passes strict screening thresholds, but this does not mean the estimate is precise or buy-ready."
            if strict_passes
            else "Does not pass strict screening thresholds; inspect the listed evidence reasons before fieldwork."
        ),
        "comparable_anchors": [
            {
                "name": anchor["name"],
                "weekly_passers": round(anchor["weekly_passers"]),
                "feature_distance": round(anchor["feature_distance"], 3),
                "reference_type": (
                    "co_located_training_anchor" if anchor["geo_distance_m"] < 1 else "comparable_anchor"
                ),
            }
            for anchor in row["comparable_anchors"]
        ],
        "runtime": runtime,
        "data_freshness": {
            "data_as_of": data_as_of,
            "age_days": data_age_days,
            "status": "historical_requires_current_check" if data_age_days is None or data_age_days > 365 else "recent",
            "note": "Verify construction, route, and local-condition changes since the data date.",
        },
        "field_validation_priorities": [
            "Run short pedestrian counts across relevant days and times.",
            "Record approach direction, screen orientation, sight lines, and obstructions.",
            "Observe dwell, crossing, queueing, and stopping behavior.",
            "Check recent construction and public-transport route changes.",
            "Join inventory, price/CPM, format, and audience data before purchase.",
        ],
        "boundary": (
            "Early-stage Amsterdam screening estimate; not live traffic, campaign ROI, "
            "inventory availability, price, demographics, or screen visibility."
        ),
    }


def evaluate_anchor(anchor: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    return evaluate_location(
        float(anchor["lat"]),
        float(anchor["lon"]),
        str(anchor["name"]),
        bundle=bundle,
    )


def run(query: str, candidate_name: str = "CLI candidate") -> dict[str, Any]:
    bundle = load_runtime_bundle()
    unsupported = unsupported_response(query, bundle)
    if unsupported:
        return unsupported
    lat_lon = extract_lat_lon(query)
    if lat_lon:
        candidate = evaluate_location(lat_lon[0], lat_lon[1], candidate_name, bundle=bundle)
        return {"kind": "location_evaluation", "candidate": public_candidate(candidate)}

    anchor = find_anchor(query, bundle["anchors"])
    asks_for_shortlist = bool(re.search(r"recommend|shortlist|rank|top |best candidate", query.lower()))
    if anchor and not asks_for_shortlist:
        candidate = evaluate_anchor(anchor, bundle)
        return {"kind": "location_evaluation", "candidate": public_candidate(candidate)}

    candidates = [evaluate_anchor(item, bundle) for item in bundle["anchors"]]
    brief = parse_campaign_brief(query)
    result = build_shortlist(candidates, brief)
    return {
        "kind": "campaign_shortlist",
        "decision_status": "screening_only",
        "boundary": "Shortlisted means priority for field validation, not a final media-buying recommendation.",
        "candidate_universe": {
            "type": "historical_training_anchors",
            "count": len(candidates),
            "is_live_zap_inventory": False,
        },
        "brief": result["brief"],
        "selected": [public_candidate(row) for row in result["selected"]],
        "field_validation_candidates": [public_candidate(row) for row in result["selected"]],
        "eligible_count": len(result["eligible"]),
        "excluded_count": len(result["excluded"]),
        "excluded": [
            {"name": item["candidate"]["name"], "reasons": item["reasons"]}
            for item in result["excluded"]
        ],
        "runtime": bundle.get("runtime", {}),
    }


def print_human(result: dict[str, Any]) -> None:
    if result["kind"] == "unsupported_decision":
        print(result["boundary"])
        print("Unsupported fields: " + ", ".join(result["unsupported_fields"]))
        print("Missing inputs: " + "; ".join(result["missing_inputs"]))
        return
    if result["kind"] == "location_evaluation":
        row = result["candidate"]
        print(row["name"])
        print(
            f"Predicted typical weekly passers: {row['predicted_weekly_passers']:,} "
            f"({row['prediction_low']:,}-{row['prediction_high']:,})"
        )
        print(
            "Reliability evidence: "
            f"similar-anchor error {row['similar_anchor_validation_error_pct']}%; "
            f"historical spread {row['historical_spread_pct']}%; "
            f"{row['outside_feature_count']} inputs outside range; "
            f"joint similarity p{row['joint_feature_similarity_percentile']}."
        )
        print(row["boundary"])
        return

    brief = result["brief"]
    print(
        f"{brief['evidence_policy'].title()} {brief['objective']} shortlist "
        f"({result['eligible_count']} eligible):"
    )
    for index, row in enumerate(result["selected"], start=1):
        print(f"{index}. {row['name']}: {row['predicted_weekly_passers']:,} typical weekly passers")
    print(result["boundary"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Amsterdam OOH candidates with the versioned runtime bundle.")
    parser.add_argument("query", nargs="*", help="Location question or campaign shortlist request")
    parser.add_argument("--json", action="store_true", help="Return structured JSON")
    parser.add_argument("--name", default="CLI candidate", help="Label for a coordinate evaluation")
    args = parser.parse_args()
    query = " ".join(args.query) or "Recommend a balanced top 5 exposure shortlist"
    result = run(query, candidate_name=args.name)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human(result)


if __name__ == "__main__":
    main()
