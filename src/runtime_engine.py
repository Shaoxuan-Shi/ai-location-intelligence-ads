from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "outputs" / "runtime" / "runtime_bundle.json"

AMSTERDAM_BOUNDS = {
    "lat_min": 52.30,
    "lat_max": 52.43,
    "lon_min": 4.75,
    "lon_max": 5.05,
}

HIGH_STREET_TYPES = {"clothes", "shoes", "jewelry", "department_store", "gift", "cosmetics", "fashion_accessories"}
LODGING_TYPES = {"hotel", "hostel", "guest_house", "apartment", "motel"}


def load_runtime_bundle(path: Path = DEFAULT_BUNDLE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_coordinate(lat: float, lon: float) -> None:
    if not (
        AMSTERDAM_BOUNDS["lat_min"] <= lat <= AMSTERDAM_BOUNDS["lat_max"]
        and AMSTERDAM_BOUNDS["lon_min"] <= lon <= AMSTERDAM_BOUNDS["lon_max"]
    ):
        raise ValueError(
            "Coordinates must be within the supported Amsterdam area "
            "(latitude 52.30-52.43, longitude 4.75-5.05)."
        )


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def percentile(value: float, values: list[float], higher_is_better: bool = True) -> float:
    if not values:
        return 50.0
    lower = sum(item < value for item in values)
    equal = sum(item == value for item in values)
    result = 100.0 * (lower + 0.5 * equal) / len(values)
    if not higher_is_better:
        result = 100.0 - result
    return round(result, 1)


def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    for index, first in enumerate(ring):
        second = ring[(index + 1) % len(ring)]
        x1, y1 = first[:2]
        x2, y2 = second[:2]
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-15) + x1:
            inside = not inside
    return inside


def _point_in_polygon(lon: float, lat: float, polygon: list[list[list[float]]]) -> bool:
    return bool(polygon and _point_in_ring(lon, lat, polygon[0]) and not any(
        _point_in_ring(lon, lat, ring) for ring in polygon[1:]
    ))


def _shopping_area(bundle: dict[str, Any], lat: float, lon: float) -> dict[str, Any]:
    for area in bundle["shoppingAreas"]:
        geometry = area.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") == "Polygon" and _point_in_polygon(lon, lat, coordinates):
            return area
        if geometry.get("type") == "MultiPolygon" and any(
            _point_in_polygon(lon, lat, polygon) for polygon in coordinates
        ):
            return area
    return {"name": "", "category": ""}


def _count_nearby(bundle: dict[str, Any], lat: float, lon: float, radius: float, category: str) -> int:
    return sum(
        point[2] == category and haversine_m(lat, lon, float(point[0]), float(point[1])) <= radius
        for point in bundle["pois"]
    )


def _count_matching(
    bundle: dict[str, Any], lat: float, lon: float, radius: float, category: str, subtypes: set[str]
) -> int:
    return sum(
        point[2] == category
        and point[3] in subtypes
        and haversine_m(lat, lon, float(point[0]), float(point[1])) <= radius
        for point in bundle["pois"]
    )


def _nearest_transit(bundle: dict[str, Any], lat: float, lon: float) -> float:
    distances = [
        haversine_m(lat, lon, float(point[0]), float(point[1]))
        for point in bundle["pois"]
        if point[2] == "transit"
    ]
    return min(distances) if distances else 9999.0


def _nearby_anchor_exposure(bundle: dict[str, Any], lat: float, lon: float) -> tuple[float, str, float]:
    weighted = 0.0
    total = 0.0
    nearest_name = ""
    nearest_distance = 9999.0
    for anchor in bundle["anchors"]:
        distance = haversine_m(lat, lon, float(anchor["lat"]), float(anchor["lon"]))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_name = str(anchor["name"])
        weight = 1.0 / max(distance, 80.0)
        weighted += float(anchor["actual"]) * weight
        total += weight
    return (weighted / total if total else 0.0), nearest_name, nearest_distance


def _transform(raw_features: dict[str, float], model: dict[str, Any]) -> list[float]:
    values = []
    for spec in model["features"]:
        value = float(raw_features.get(spec["name"], 0.0))
        values.append(value if spec.get("transform") == "identity" else math.log1p(max(value, 0.0)))
    return values


def _standardize(values: list[float], model: dict[str, Any]) -> list[float]:
    return [
        (value - float(spec["mean"])) / (float(spec["std"]) or 1.0)
        for value, spec in zip(values, model["features"])
    ]


def _feature_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def evaluate_location(
    lat: float,
    lon: float,
    name: str = "Custom candidate",
    *,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_coordinate(lat, lon)
    context = bundle or load_runtime_bundle()
    model = context["model"]
    area = _shopping_area(context, lat, lon)
    nearby_exposure, nearest_anchor_name, nearest_anchor_distance = _nearby_anchor_exposure(context, lat, lon)

    retail = _count_nearby(context, lat, lon, 80, "shop")
    horeca = _count_nearby(context, lat, lon, 120, "horeca")
    transit = _count_nearby(context, lat, lon, 250, "transit")
    tourism = _count_nearby(context, lat, lon, 300, "tourism")
    nearest_transit = round(_nearest_transit(context, lat, lon), 1)
    raw_features = {
        "shop_count_80m": float(retail),
        "horeca_count_120m": float(horeca),
        "transit_stop_count_250m": float(transit),
        "distance_to_nearest_transit_m": nearest_transit,
        "tourism_poi_count_300m": float(tourism),
        "in_official_shopping_area": float(bool(area.get("name"))),
        "high_street_shop_count_120m": float(
            _count_matching(context, lat, lon, 120, "shop", HIGH_STREET_TYPES)
        ),
        "in_key_shopping_area": float(area.get("category") == "K"),
        "lodging_count_500m": float(_count_matching(context, lat, lon, 500, "tourism", LODGING_TYPES)),
        "horeca_count_200m": float(_count_nearby(context, lat, lon, 200, "horeca")),
        "nearby_anchor_median_passers": float(round(nearby_exposure)),
    }
    x_std = _standardize(_transform(raw_features, model), model)
    pred_log = float(model["coefficients"][0]) + sum(
        value * float(coefficient) for value, coefficient in zip(x_std, model["coefficients"][1:])
    )
    novelty = max(0.0, max(abs(value) for value in x_std) - 2.0)
    interval_scale = float(model["intervalLogMargin"]) * (1.0 + 0.25 * novelty)
    predicted = max(math.expm1(pred_log), 0.0)
    low = max(math.expm1(pred_log - interval_scale), 0.0)
    high = max(math.expm1(pred_log + interval_scale), 0.0)

    comparables = []
    for anchor in context["anchors"]:
        anchor_std = _standardize(_transform(anchor["rawFeatures"], model), model)
        comparables.append(
            {
                "name": anchor["name"],
                "weekly_passers": anchor["actual"],
                "feature_distance": _feature_distance(x_std, anchor_std),
                "geo_distance_m": haversine_m(lat, lon, float(anchor["lat"]), float(anchor["lon"])),
                "loo_error": float(anchor.get("looPctError") or 0.75),
                "variability": float(anchor.get("variability") or 0.25),
            }
        )
    comparables.sort(key=lambda row: row["feature_distance"])
    closest = comparables[:3]
    comparable_error = sum(row["loo_error"] for row in closest) / len(closest)
    variability = sum(row["variability"] for row in closest) / len(closest)
    ranges = context["modelFeatureRanges"]
    outside = [
        spec["name"]
        for spec in model["features"]
        if raw_features.get(spec["name"], 0.0) < float(ranges[spec["name"]]["min"])
        or raw_features.get(spec["name"], 0.0) > float(ranges[spec["name"]]["max"])
    ]
    support_threshold = float(context["featureSupport"]["typicalThreshold"])
    support_count = sum(row["feature_distance"] <= support_threshold for row in comparables)
    joint_similarity = percentile(
        comparables[0]["feature_distance"],
        [float(value) for value in context["featureSupport"]["nearestDistances"]],
        higher_is_better=False,
    )
    training = context["trainingValues"]
    return {
        "name": name,
        "lat": lat,
        "lon": lon,
        "predicted_weekly_passers": predicted,
        "prediction_low": low,
        "prediction_high": high,
        "interval_coverage": "approximate_80_percent",
        "interval_method": model.get("intervalMethod", "80th percentile absolute LOOCV log residual"),
        "interval_width_ratio": (high - low) / max(predicted, 1.0),
        "retail_percentile": percentile(retail, training["shop_count_80m"]),
        "horeca_percentile": percentile(horeca, training["horeca_count_120m"]),
        "transit_percentile": percentile(transit, training["transit_stop_count_250m"]),
        "tourism_percentile": percentile(tourism, training["tourism_poi_count_300m"]),
        "similar_anchor_validation_error": comparable_error,
        "comparable_anchor_variability": variability,
        "outside_feature_count": len(outside),
        "outside_features": outside,
        "feature_support_count": support_count,
        "feature_support_threshold": support_threshold,
        "joint_feature_similarity_percentile": joint_similarity,
        "feature_mean_abs_z": sum(abs(value) for value in x_std) / len(x_std),
        "feature_max_abs_z": max(abs(value) for value in x_std),
        "shopping_area_name": area.get("name", ""),
        "shopping_area_category": area.get("category", ""),
        "nearest_anchor_name": nearest_anchor_name,
        "nearest_anchor_distance_m": nearest_anchor_distance,
        "raw_features": raw_features,
        "comparable_anchors": closest,
        "runtime": context.get("runtime", {}),
    }
