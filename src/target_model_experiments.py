from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from model_weekly_passers import (
    RAW_FEATURES,
    add_intercept,
    gaussian_solve,
    metrics,
    predict_linear,
    rank_correlation,
    standardize_apply,
    standardize_fit,
    transform_row,
    transpose,
)
from utils import DATA_PROCESSED, DATA_RAW, OUTPUT_TABLES, expm1, log1p, median, quantile, read_csv, safe_float, write_csv


ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
PRESET_PARAMETERS = {
    "huber_ridge": (3.0, 1.35),
    "stability_weighted_ridge": (3.0, 1.0),
}
EXACT_COORDINATE_DECIMALS = 6


def fit_weighted_ridge(
    x: list[list[float]], y: list[float], alpha: float, weights: list[float]
) -> list[float]:
    x_i = add_intercept(x)
    xt = transpose(x_i)
    xtwx = [
        [sum(weights[i] * left[i] * right[i] for i in range(len(y))) for right in xt]
        for left in xt
    ]
    for idx in range(1, len(xtwx)):
        xtwx[idx][idx] += alpha
    xtwy = [sum(weights[i] * col[i] * y[i] for i in range(len(y))) for col in xt]
    return gaussian_solve(xtwx, xtwy)


def fit_huber_ridge(
    x: list[list[float]], y: list[float], alpha: float, delta: float, iterations: int = 30
) -> list[float]:
    weights = [1.0] * len(y)
    coefficients = fit_weighted_ridge(x, y, alpha, weights)
    for _ in range(iterations):
        residuals = [actual - predicted for actual, predicted in zip(y, predict_linear(x, coefficients))]
        scale = median([abs(value - median(residuals)) for value in residuals]) / 0.6745
        scale = max(scale, 1e-6)
        threshold = delta * scale
        updated_weights = [1.0 if abs(value) <= threshold else threshold / abs(value) for value in residuals]
        updated = fit_weighted_ridge(x, y, alpha, updated_weights)
        if max(abs(a - b) for a, b in zip(coefficients, updated)) < 1e-6:
            coefficients = updated
            break
        coefficients = updated
        weights = updated_weights
    return coefficients


def stability_weights(rows: list[dict[str, Any]], power: float) -> list[float]:
    raw = [1.0 / (1.0 + safe_float(row.get("variability_ratio"))) ** power for row in rows]
    scale = len(raw) / sum(raw)
    return [value * scale for value in raw]


def fit_predict(
    model: str,
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    alpha: float,
    extra: float = 0.0,
) -> list[float]:
    train_raw = [transform_row(row) for row in train_rows]
    test_raw = [transform_row(row) for row in test_rows]
    train_x, specs = standardize_fit(train_raw, RAW_FEATURES)
    test_x = standardize_apply(test_raw, specs)
    train_y = [log1p(safe_float(row["weekly_passers"])) for row in train_rows]
    if model == "ridge":
        coefficients = fit_weighted_ridge(train_x, train_y, alpha, [1.0] * len(train_y))
    elif model == "huber_ridge":
        coefficients = fit_huber_ridge(train_x, train_y, alpha, extra)
    elif model == "stability_weighted_ridge":
        coefficients = fit_weighted_ridge(train_x, train_y, alpha, stability_weights(train_rows, extra))
    else:
        raise ValueError(model)
    return predict_linear(test_x, coefficients)


def parameter_grid(model: str) -> list[tuple[float, float]]:
    if model == "ridge":
        return [(alpha, 0.0) for alpha in ALPHAS]
    if model in PRESET_PARAMETERS:
        return [PRESET_PARAMETERS[model]]
    raise ValueError(model)


def plain_loocv(rows: list[dict[str, Any]], model: str, alpha: float, extra: float) -> list[float]:
    predictions = []
    for holdout in range(len(rows)):
        train = [row for idx, row in enumerate(rows) if idx != holdout]
        predictions.extend(fit_predict(model, train, [rows[holdout]], alpha, extra))
    return predictions


def choose_parameters(rows: list[dict[str, Any]], model: str) -> tuple[float, float]:
    actual = [log1p(safe_float(row["weekly_passers"])) for row in rows]
    best = (float("inf"), 0.0, 0.0)
    for alpha, extra in parameter_grid(model):
        predicted = plain_loocv(rows, model, alpha, extra)
        score = metrics(actual, predicted)["rmse_log"]
        best = min(best, (score, alpha, extra))
    return best[1], best[2]


def nested_loocv(rows: list[dict[str, Any]], model: str) -> tuple[list[float], list[tuple[float, float]]]:
    if model in PRESET_PARAMETERS:
        alpha, extra = PRESET_PARAMETERS[model]
        return plain_loocv(rows, model, alpha, extra), [(alpha, extra)] * len(rows)
    predictions = []
    selected = []
    for holdout in range(len(rows)):
        train = [row for idx, row in enumerate(rows) if idx != holdout]
        alpha, extra = choose_parameters(train, model)
        predictions.extend(fit_predict(model, train, [rows[holdout]], alpha, extra))
        selected.append((alpha, extra))
    return predictions, selected


def parse_day(value: str) -> date:
    return date.fromisoformat(value[:10])


def weekly_history() -> dict[str, dict[date, float]]:
    grouped: dict[str, dict[date, float]] = defaultdict(dict)
    for row in read_csv(DATA_RAW / "crowdmonitor_weekly_history.csv"):
        count = safe_float(row.get("aantalPassanten"))
        if count > 0:
            grouped[row["naamLocatie"]][parse_day(row["datumUur"])] = count
    return grouped


def expand_directional_rows(
    feature_rows: list[dict[str, Any]], history: dict[str, dict[date, float]]
) -> list[dict[str, Any]]:
    expanded = []
    for row in feature_rows:
        names = row.get("source_location_names", row["location_name"]).split(" | ")
        for name in names:
            item = dict(row)
            counts = list(history[name].values())
            p25 = quantile(counts, 0.25)
            p75 = quantile(counts, 0.75)
            item["location_name"] = name
            item["weekly_passers"] = round(median(counts))
            item["variability_ratio"] = (p75 - p25) / max(median(counts), 1.0)
            expanded.append(item)
    return expanded


def citywide_week_factors(history: dict[str, dict[date, float]]) -> dict[date, float]:
    location_medians = {name: median(list(values.values())) for name, values in history.items()}
    by_week: dict[date, list[float]] = defaultdict(list)
    for name, values in history.items():
        for week, count in values.items():
            if location_medians[name] > 0:
                by_week[week].append(count / location_medians[name])
    return {week: median(ratios) for week, ratios in by_week.items()}


def normal_counts(
    values: dict[date, float], week_factors: dict[date, float], excluded_weeks: set[date] | None = None
) -> tuple[list[float], int]:
    counts = list(values.values())
    center = median(counts)
    excluded_weeks = excluded_weeks or set()
    kept = []
    rejected = 0
    for week, count in values.items():
        sensor_failure = count < 0.20 * center or count > 5.0 * center
        citywide_incomplete = week_factors.get(week, 1.0) < 0.60
        if sensor_failure or citywide_incomplete or week in excluded_weeks:
            rejected += 1
        else:
            kept.append(count)
    return kept or counts, rejected


def coordinate_key(row: dict[str, Any]) -> tuple[float, float]:
    return (
        round(safe_float(row["lat"]), EXACT_COORDINATE_DECIMALS),
        round(safe_float(row["lon"]), EXACT_COORDINATE_DECIMALS),
    )


def physical_site_rows(
    feature_rows: list[dict[str, Any]],
    history: dict[str, dict[date, float]],
    *,
    clean_anomalies: bool,
    roadwork_weeks: dict[str, set[date]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_coordinate: dict[tuple[float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_coordinate[coordinate_key(row)].append(row)
    week_factors = citywide_week_factors(history)
    result = []
    audit = []
    for group in by_coordinate.values():
        names = [row["location_name"] for row in group]
        common_weeks = set.intersection(*(set(history[name]) for name in names))
        site_history = {
            week: sum(history[name][week] for name in names) / len(names)
            for week in common_weeks
        }
        excluded = set()
        if roadwork_weeks:
            for name in names:
                excluded.update(roadwork_weeks.get(name, set()))
        if clean_anomalies:
            counts, rejected = normal_counts(site_history, week_factors, excluded)
        else:
            counts = list(site_history.values())
            rejected = 0
        row = dict(group[0])
        row["location_name"] = " + ".join(names)
        row["sensor"] = " + ".join(item.get("sensor", "") for item in group)
        row["weekly_passers"] = round(median(counts))
        row["mean_weekly_passers"] = round(sum(counts) / len(counts))
        row["p25_weekly_passers"] = round(quantile(counts, 0.25))
        row["p75_weekly_passers"] = round(quantile(counts, 0.75))
        row["iqr_weekly_passers"] = row["p75_weekly_passers"] - row["p25_weekly_passers"]
        row["variability_ratio"] = row["iqr_weekly_passers"] / max(row["weekly_passers"], 1)
        row["weekly_observation_count"] = len(counts)
        row["target_definition"] = (
            "co_located_direction_neutral_normal_week_median"
            if clean_anomalies
            else "co_located_direction_neutral_weekly_median"
        )
        result.append(row)
        audit.append(
            {
                "location_name": row["location_name"],
                "source_sensor_count": len(group),
                "raw_week_count": len(site_history),
                "excluded_week_count": rejected,
                "target_week_count": len(counts),
                "raw_median": round(median(list(site_history.values()))),
                "target_median": row["weekly_passers"],
            }
        )
    return result, audit


def geometry_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    coordinates = geometry.get("coordinates", [])
    geometry_type = geometry.get("type")
    if geometry_type == "Polygon":
        return coordinates
    if geometry_type == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    if geometry_type == "LineString":
        return [coordinates]
    if geometry_type == "MultiLineString":
        return coordinates
    if geometry_type == "Point":
        return [[coordinates]]
    if geometry_type == "MultiPoint":
        return [[point] for point in coordinates]
    return []


def distance_to_segment_m(lat: float, lon: float, a: list[float], b: list[float]) -> float:
    x_scale = 111_320.0 * math.cos(math.radians(lat))
    y_scale = 110_540.0
    ax, ay = (a[0] - lon) * x_scale, (a[1] - lat) * y_scale
    bx, by = (b[0] - lon) * x_scale, (b[1] - lat) * y_scale
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    position = 0.0 if denominator == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / denominator))
    return math.hypot(ax + position * dx, ay + position * dy)


def distance_to_geometry_m(geometry: dict[str, Any], lat: float, lon: float) -> float:
    closest = float("inf")
    for ring in geometry_rings(geometry):
        if len(ring) == 1:
            pairs = [(ring[0], ring[0])]
        else:
            pairs = list(zip(ring, ring[1:] + ring[:1]))
        for start, end in pairs:
            closest = min(closest, distance_to_segment_m(lat, lon, start, end))
    return closest


def nearby_short_roadwork_weeks(feature_rows: list[dict[str, Any]]) -> dict[str, set[date]]:
    path = DATA_RAW / "wior_2023_2024.geojson"
    if not path.exists():
        return {}
    features = json.loads(path.read_text(encoding="utf-8")).get("features", [])
    result: dict[str, set[date]] = defaultdict(set)
    for feature in features:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry")
        start_value = properties.get("datumStartUitvoering")
        end_value = properties.get("datumEindeUitvoering")
        if not geometry or not start_value or not end_value:
            continue
        start, end = parse_day(start_value), parse_day(end_value)
        if end < start or (end - start).days > 60:
            continue
        for row in feature_rows:
            if distance_to_geometry_m(geometry, safe_float(row["lat"]), safe_float(row["lon"])) > 50:
                continue
            first_week = start - timedelta(days=start.weekday())
            last_week = end - timedelta(days=end.weekday())
            week = first_week
            while week <= last_week:
                result[row["location_name"]].add(week)
                week += timedelta(days=7)
    return result


def evaluate_variant(label: str, rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    actual_log = [log1p(safe_float(row["weekly_passers"])) for row in rows]
    predicted_log, selected = nested_loocv(rows, model)
    score = metrics(actual_log, predicted_log)
    actual = [expm1(value) for value in actual_log]
    predicted = [expm1(value) for value in predicted_log]
    selected_counts: dict[str, int] = defaultdict(int)
    for alpha, extra in selected:
        selected_counts[f"{alpha:g}/{extra:g}"] += 1
    common = max(selected_counts, key=selected_counts.get)
    return {
        "target_variant": label,
        "model": model,
        "anchors": len(rows),
        **{key: round(value, 4) for key, value in score.items()},
        "rank_correlation": round(rank_correlation(actual, predicted), 4),
        "most_selected_alpha_extra": common,
        "selection_frequency": selected_counts[common],
    }


def main() -> None:
    feature_rows = [row for row in read_csv(DATA_PROCESSED / "site_features_weekly.csv") if safe_float(row.get("weekly_passers")) > 0]
    history = weekly_history()
    directional_rows = expand_directional_rows(feature_rows, history)
    roadwork_weeks = nearby_short_roadwork_weeks(directional_rows)

    physical_rows, physical_audit = physical_site_rows(directional_rows, history, clean_anomalies=False)
    clean_rows, clean_audit = physical_site_rows(directional_rows, history, clean_anomalies=True)
    roadwork_rows, roadwork_audit = physical_site_rows(
        directional_rows, history, clean_anomalies=True, roadwork_weeks=roadwork_weeks
    )

    variants = [
        ("current_directional_sensor_medians", directional_rows),
        ("direction_neutral_co_located_medians", physical_rows),
        ("direction_neutral_normal_week_medians", clean_rows),
        ("direction_neutral_normal_week_minus_short_nearby_roadworks", roadwork_rows),
    ]
    results = []
    for label, rows in variants:
        for model in ["ridge", "huber_ridge", "stability_weighted_ridge"]:
            results.append(evaluate_variant(label, rows, model))
    results.sort(key=lambda row: (safe_float(row["rmse_log"]), safe_float(row["mape_pct"])))
    write_csv(OUTPUT_TABLES / "target_model_experiment_results.csv", results)
    write_csv(OUTPUT_TABLES / "target_audit_physical.csv", physical_audit)
    write_csv(OUTPUT_TABLES / "target_audit_normal_week.csv", clean_audit)
    write_csv(OUTPUT_TABLES / "target_audit_roadworks.csv", roadwork_audit)
    print("Target/model experiments complete")
    for row in results:
        print(
            f"{row['rmse_log']:.4f} RMSE | {row['mape_pct']:.1f}% MAPE | "
            f"{row['rank_correlation']:.3f} rank | {row['target_variant']} | {row['model']}"
        )


if __name__ == "__main__":
    main()
