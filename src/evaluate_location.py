from __future__ import annotations

import json
import math
import sys
from typing import Any

from build_features import (
    HIGH_STREET_TYPES,
    LODGING_TYPES,
    count_matching,
    count_nearby,
    load_horeca_points,
    load_osm_points,
    nearest_distance,
    shopping_area_match,
)
from model_weekly_passers import RAW_FEATURES, predict_linear, transform_row
from utils import (
    DATA_PROCESSED,
    DATA_RAW,
    OUTPUT_TABLES,
    expm1,
    haversine_m,
    log1p,
    read_csv,
    read_json,
    safe_float,
    quantile,
    write_json,
)


AMSTERDAM_BOUNDS = {
    "lat_min": 52.30,
    "lat_max": 52.43,
    "lon_min": 4.75,
    "lon_max": 5.05,
}


def validate_candidate_coordinates(lat: float, lon: float) -> None:
    """Reject coordinates outside the product's supported Amsterdam area."""
    if not (
        AMSTERDAM_BOUNDS["lat_min"] <= lat <= AMSTERDAM_BOUNDS["lat_max"]
        and AMSTERDAM_BOUNDS["lon_min"] <= lon <= AMSTERDAM_BOUNDS["lon_max"]
    ):
        raise ValueError(
            "Coordinates must be within the supported Amsterdam area "
            "(latitude 52.30-52.43, longitude 4.75-5.05)."
        )


def percentile_against_training(value: float, values: list[float], higher_is_better: bool = True) -> float:
    if not values:
        return 50.0
    lower = sum(1 for item in values if item < value)
    equal = sum(1 for item in values if item == value)
    percentile = 100.0 * (lower + 0.5 * equal) / len(values)
    if not higher_is_better:
        percentile = 100.0 - percentile
    return round(percentile, 1)


def raw_feature_dict(row: dict[str, Any]) -> dict[str, float]:
    return {name: safe_float(row.get(name)) for name in RAW_FEATURES}


def standardize_vector(values: list[float], artifact: dict[str, Any]) -> list[float]:
    output = []
    for value, spec in zip(values, artifact["features"]):
        std = safe_float(spec["std"], 1.0) or 1.0
        output.append((value - safe_float(spec["mean"])) / std)
    return output


def feature_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 999.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def nearest_comparable_anchors(x_std: list[float], artifact: dict[str, Any], k: int = 3) -> list[dict[str, Any]]:
    anchors = []
    for row in artifact["training_rows"]:
        transformed = [
            safe_float(row["raw_features"].get(spec["name"]))
            if spec.get("transform") == "identity"
            else log1p(safe_float(row["raw_features"].get(spec["name"])))
            for spec in artifact["features"]
        ]
        anchor_std = standardize_vector(transformed, artifact)
        anchors.append(
            {
                "location_name": row["location_name"],
                "lat": row["lat"],
                "lon": row["lon"],
                "weekly_passers": row["weekly_passers"],
                "variability_ratio": row.get("variability_ratio", 0.0),
                "feature_distance": feature_distance(x_std, anchor_std),
            }
        )
    anchors.sort(key=lambda row: row["feature_distance"])
    return anchors[:k]


def feature_support_reference(artifact: dict[str, Any]) -> tuple[list[float], float]:
    vectors = []
    for row in artifact["training_rows"]:
        transformed = [
            safe_float(row["raw_features"].get(spec["name"]))
            if spec.get("transform") == "identity"
            else log1p(safe_float(row["raw_features"].get(spec["name"])))
            for spec in artifact["features"]
        ]
        vectors.append(standardize_vector(transformed, artifact))
    nearest_distances = [
        min(feature_distance(vector, other) for other_index, other in enumerate(vectors) if other_index != index)
        for index, vector in enumerate(vectors)
    ]
    return nearest_distances, quantile(nearest_distances, 0.75)


def build_candidate_features(name: str, lat: float, lon: float) -> dict[str, Any]:
    winkelgebieden = read_json(DATA_RAW / "winkelgebieden.geojson")
    horeca_points = load_horeca_points()
    osm_points = load_osm_points()
    shop_points = [point for point in osm_points if point["category"] == "shop"]
    tourism_points = [point for point in osm_points if point["category"] == "tourism"]
    in_area, area_name, area_category = shopping_area_match(lon, lat, winkelgebieden)
    row: dict[str, Any] = {
        "location_name": name,
        "lat": lat,
        "lon": lon,
        "shop_count_80m": count_nearby(osm_points, lat, lon, 80, "shop"),
        "shop_count_120m": count_nearby(osm_points, lat, lon, 120, "shop"),
        "horeca_count_120m": count_nearby(horeca_points, lat, lon, 120),
        "horeca_count_200m": count_nearby(horeca_points, lat, lon, 200),
        "transit_stop_count_250m": count_nearby(osm_points, lat, lon, 250, "transit"),
        "distance_to_nearest_transit_m": round(nearest_distance(osm_points, lat, lon, "transit"), 1),
        "tourism_poi_count_300m": count_nearby(osm_points, lat, lon, 300, "tourism"),
        "high_street_shop_count_120m": count_matching(
            shop_points, lat, lon, 120, lambda point: point.get("shop_type") in HIGH_STREET_TYPES
        ),
        "lodging_count_500m": count_matching(
            tourism_points, lat, lon, 500, lambda point: point.get("tourism_type") in LODGING_TYPES
        ),
        "in_official_shopping_area": in_area,
        "in_key_shopping_area": int(area_category == "K"),
        "shopping_area_name": area_name,
        "shopping_area_category": area_category,
    }
    artifact = read_json(OUTPUT_TABLES / "model_artifact.json")
    weighted_sum = 0.0
    weight_total = 0.0
    nearest_anchor_distance = 9999.0
    nearest_name = ""
    for anchor in artifact["training_rows"]:
        distance = haversine_m(lat, lon, safe_float(anchor["lat"]), safe_float(anchor["lon"]))
        if distance < nearest_anchor_distance:
            nearest_anchor_distance = distance
            nearest_name = anchor["location_name"]
        weight = 1.0 / max(distance, 80.0)
        weighted_sum += safe_float(anchor["weekly_passers"]) * weight
        weight_total += weight
    row["nearby_anchor_median_passers"] = round(weighted_sum / weight_total if weight_total else 0.0)
    row["nearest_anchor_distance_m"] = round(nearest_anchor_distance, 1)
    row["nearest_anchor_name"] = nearest_name

    training = read_csv(DATA_PROCESSED / "site_features_weekly.csv")
    percentile_specs = [
        ("shop_count_80m", True),
        ("shop_count_120m", True),
        ("horeca_count_120m", True),
        ("transit_stop_count_250m", True),
        ("distance_to_nearest_transit_m", False),
        ("tourism_poi_count_300m", True),
        ("nearby_anchor_median_passers", True),
    ]
    for column, higher_is_better in percentile_specs:
        if column == "nearby_anchor_median_passers":
            values = [safe_float(item["raw_features"].get(column)) for item in artifact["training_rows"]]
        else:
            values = [safe_float(item[column]) for item in training]
        row[f"{column}_percentile"] = percentile_against_training(safe_float(row[column]), values, higher_is_better)
    return row


def evaluate_candidate(name: str, lat: float, lon: float) -> dict[str, Any]:
    validate_candidate_coordinates(lat, lon)
    artifact = read_json(OUTPUT_TABLES / "model_artifact.json")
    row = build_candidate_features(name, lat, lon)
    x_raw = transform_row(row)
    x_std = standardize_vector(x_raw, artifact)
    pred_log = predict_linear([x_std], artifact["coefficients"])[0]
    novelty = max(0.0, max(abs(value) for value in x_std) - 2.0)
    interval_scale = safe_float(artifact.get("interval_log_margin", artifact["rmse_log_fit"])) * (1.0 + 0.25 * novelty)
    point = expm1(pred_log)
    low = expm1(pred_log - interval_scale)
    high = expm1(pred_log + interval_scale)
    width_ratio = (high - low) / max(point, 1.0)
    all_comparables = nearest_comparable_anchors(x_std, artifact, k=len(artifact["training_rows"]))
    comparables = all_comparables[:3]
    support_reference, support_threshold = feature_support_reference(artifact)
    support_count = sum(anchor["feature_distance"] <= support_threshold for anchor in all_comparables)
    joint_similarity_percentile = percentile_against_training(
        all_comparables[0]["feature_distance"], support_reference, higher_is_better=False
    )
    comparable_error = 0.75
    comparable_variability = 0.25
    validation_rows = read_csv(OUTPUT_TABLES / "site_predictions_weekly.csv")
    validation_lookup = {item["location_name"]: item for item in validation_rows}
    comparable_errors = [
        safe_float(validation_lookup.get(anchor["location_name"], {}).get("loocv_absolute_percentage_error")) / 100
        for anchor in comparables
        if anchor["location_name"] in validation_lookup
    ]
    if comparable_errors:
        comparable_error = sum(comparable_errors) / len(comparable_errors)
    comparable_variabilities = [safe_float(anchor.get("variability_ratio")) for anchor in comparables]
    if comparable_variabilities:
        comparable_variability = sum(comparable_variabilities) / len(comparable_variabilities)

    mean_abs_z = sum(abs(value) for value in x_std) / len(x_std)
    max_abs_z = max(abs(value) for value in x_std)
    outside_feature_count = 0
    for feature_name in RAW_FEATURES:
        training_values = [safe_float(anchor["raw_features"].get(feature_name)) for anchor in artifact["training_rows"]]
        value = safe_float(row.get(feature_name))
        if value < min(training_values) or value > max(training_values):
            outside_feature_count += 1
    row.update(
        {
            "predicted_weekly_passers": round(point),
            "prediction_low": round(low),
            "prediction_high": round(high),
            "interval_width_ratio": round(width_ratio, 3),
            "feature_mean_abs_z": round(mean_abs_z, 3),
            "feature_max_abs_z": round(max_abs_z, 3),
            "outside_feature_count": outside_feature_count,
            "feature_support_count": support_count,
            "feature_support_threshold": round(support_threshold, 3),
            "joint_feature_similarity_percentile": joint_similarity_percentile,
            "similar_anchor_validation_error": round(comparable_error * 100, 1),
            "comparable_anchor_variability": round(comparable_variability, 3),
            "comparable_anchors": [
                {
                    "location_name": anchor["location_name"],
                    "weekly_passers": round(anchor["weekly_passers"]),
                    "feature_distance": round(anchor["feature_distance"], 3),
                    "geo_distance_m": round(
                        haversine_m(lat, lon, safe_float(anchor["lat"]), safe_float(anchor["lon"])), 1
                    ),
                }
                for anchor in comparables
            ],
        }
    )
    return row


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python3 src/evaluate_location.py <lat> <lon> [name]")
        raise SystemExit(2)
    lat = safe_float(sys.argv[1])
    lon = safe_float(sys.argv[2])
    name = " ".join(sys.argv[3:]) or "Custom candidate"
    try:
        result = evaluate_candidate(name, lat, lon)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    write_json(OUTPUT_TABLES / "last_candidate_evaluation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
