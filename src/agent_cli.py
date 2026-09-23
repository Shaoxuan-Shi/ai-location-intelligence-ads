from __future__ import annotations

import re
import sys
from typing import Any

from evaluate_location import evaluate_candidate
from utils import OUTPUT_TABLES, read_csv, safe_float


def objective_from_query(query: str) -> str:
    q = query.lower()
    if any(word in q for word in ["shopping", "retail", "shop", "store", "fashion"]):
        return "shopping"
    if any(word in q for word in ["tourist", "tourism", "visitor", "museum", "hotel"]):
        return "tourism"
    if any(word in q for word in ["commuter", "transit", "station", "metro", "tram"]):
        return "commuter"
    return "general"


def score(row: dict[str, Any], objective: str) -> float:
    predicted = safe_float(row["predicted_weekly_passers"])
    if objective == "shopping":
        return predicted * (0.70 + safe_float(row["shop_count_80m_percentile"]) / 250 + safe_float(row["horeca_count_120m_percentile"]) / 500)
    if objective == "tourism":
        return predicted * (0.70 + safe_float(row["tourism_poi_count_300m_percentile"]) / 220 + safe_float(row["shop_count_80m_percentile"]) / 600)
    if objective == "commuter":
        return predicted * (0.70 + safe_float(row["transit_stop_count_250m_percentile"]) / 220)
    return predicted


def row_value(row: dict[str, Any], key: str) -> float:
    return safe_float(row.get(key))


def evidence(row: dict[str, Any], objective: str = "general") -> str:
    parts = []
    if row_value(row, "shop_count_80m_percentile") >= 75:
        parts.append("strong retail-density percentile")
    if row_value(row, "horeca_count_120m_percentile") >= 75:
        parts.append("strong horeca context")
    if row_value(row, "transit_stop_count_250m_percentile") >= 75:
        parts.append("strong transit accessibility")
    if row_value(row, "tourism_poi_count_300m_percentile") >= 75:
        parts.append("strong tourism context")
    area = row.get("shopping_area_name", "")
    if area:
        parts.append(f"inside official shopping area: {area}")
    if not parts:
        parts.append("no single contextual signal is top-quartile against current anchors")
    if objective != "general":
        parts.append(f"matches {objective} objective")
    return "; ".join(parts)


def print_location_answer(row: dict[str, Any], name: str | None = None) -> None:
    title = name or row.get("location_name", "Candidate location")
    predicted = round(row_value(row, "predicted_weekly_passers"))
    low = round(row_value(row, "prediction_low"))
    high = round(row_value(row, "prediction_high"))
    width_ratio = row_value(row, "interval_width_ratio")
    similar_error = row_value(row, "similar_anchor_validation_error")
    if not similar_error:
        similar_error = row_value(row, "loocv_absolute_percentage_error")
    variability = row_value(row, "comparable_anchor_variability")
    if not variability:
        variability = row_value(row, "variability_ratio")
    outside_count = round(row_value(row, "outside_feature_count"))
    support_count = round(row_value(row, "feature_support_count"))
    support_threshold = row_value(row, "feature_support_threshold")
    joint_similarity = row_value(row, "joint_feature_similarity_percentile")
    print(title)
    print(f"Predicted typical weekly passers: {predicted:,} ({low:,}-{high:,})")
    print("Reliability evidence:")
    if row.get("comparable_anchors"):
        nearest_feature_distance = row_value(row["comparable_anchors"][0], "feature_distance")
        print(
            f"  Comparable-anchor support: nearest feature distance {nearest_feature_distance:.2f}; "
            f"{support_count} anchors within the typical threshold {support_threshold:.2f}"
        )
    print(f"  Similar-anchor held-out error: {similar_error:.1f}%")
    print(f"  Historical weekly spread: {variability * 100:.0f}% of median")
    print(
        f"  Training range check: {outside_count} model inputs outside range; "
        f"joint similarity stronger than {joint_similarity:.0f}% of training nearest matches"
    )
    print(f"Evidence: {evidence(row)}")
    if "comparable_anchors" in row:
        comparable = ", ".join(
            f"{item['location_name']} ({round(item['weekly_passers']):,})" for item in row["comparable_anchors"]
        )
        print(f"Comparable anchors: {comparable}")
    elif row.get("loocv_absolute_percentage_error"):
        print(f"LOOCV error for this anchor: {row['loocv_absolute_percentage_error']}%")
    if width_ratio > 1.5 or similar_error > 75 or variability > 0.45 or outside_count > 0:
        print("Recommendation: use for early shortlisting only; field validation is strongly recommended.")
    else:
        print("Recommendation: suitable for shortlisting; validate before final media buying.")


def extract_lat_lon(query: str) -> tuple[float, float] | None:
    numbers = [float(item) for item in re.findall(r"-?\d+\.\d+", query)]
    for i in range(len(numbers) - 1):
        lat, lon = numbers[i], numbers[i + 1]
        if 52.0 <= lat <= 52.6 and 4.5 <= lon <= 5.3:
            return lat, lon
    return None


def find_anchor(query: str, rows: list[dict[str, str]]) -> dict[str, str] | None:
    q = query.lower()
    exact = [row for row in rows if row["location_name"].lower() in q]
    if exact:
        return max(exact, key=lambda row: len(row["location_name"]))
    tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", q) if len(token) >= 4]
    for row in rows:
        name = row["location_name"].lower()
        if any(token in name for token in tokens):
            return row
    return None


def print_recommendations(query: str, rows: list[dict[str, str]]) -> None:
    objective = objective_from_query(query)
    ranked = sorted(rows, key=lambda row: score(row, objective), reverse=True)
    print(f"Objective detected: {objective}")
    print("Shortlist recommendation based on model predictions and campaign-fit signals.\n")
    for idx, row in enumerate(ranked[:5], start=1):
        print(f"{idx}. {row['location_name']}")
        print_location_answer(row)
        print()
    print("Important: this assistant estimates early-stage exposure potential, not final campaign ROI.")


def main() -> None:
    query = " ".join(sys.argv[1:]) or "Recommend 5 locations for a general campaign"
    rows = read_csv(OUTPUT_TABLES / "site_predictions_weekly.csv")
    lat_lon = extract_lat_lon(query)
    if lat_lon:
        result = evaluate_candidate("CLI candidate", lat_lon[0], lat_lon[1])
        print_location_answer(result, "CLI candidate")
        return
    anchor = find_anchor(query, rows)
    if anchor and not any(word in query.lower() for word in ["recommend", "shortlist", "top "]):
        print_location_answer(anchor)
        return
    print_recommendations(query, rows)


if __name__ == "__main__":
    main()
