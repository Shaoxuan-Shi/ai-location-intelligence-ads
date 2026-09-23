from __future__ import annotations

from typing import Any

from model_weekly_passers import (
    leave_one_out_predictions,
    log1p,
    metrics,
    predict_linear,
    rank_correlation,
    safe_float,
    standardize_apply,
    standardize_fit,
    fit_ridge,
)
from utils import DATA_PROCESSED, OUTPUT_TABLES, read_csv, write_csv


BASE = [
    "shop_count_80m",
    "horeca_count_120m",
    "transit_stop_count_250m",
    "distance_to_nearest_transit_m",
    "tourism_poi_count_300m",
    "in_official_shopping_area",
]

GROUPS = {
    "baseline": BASE,
    "retail_scales": BASE + ["shop_count_40m", "shop_count_120m", "shop_count_200m", "shop_count_300m", "high_street_shop_count_120m", "distance_to_nearest_shop_m", "retail_concentration_80_300"],
    "horeca_scales": BASE + ["horeca_count_60m", "horeca_count_200m", "horeca_count_300m", "nightlife_count_300m", "distance_to_nearest_horeca_m", "horeca_concentration_120_300"],
    "transit_hierarchy": BASE + ["transit_stop_count_100m", "transit_stop_count_500m", "high_capacity_transit_count_500m"],
    "visitor_destinations": BASE + ["tourism_poi_count_100m", "tourism_poi_count_500m", "distance_to_nearest_tourism_poi_m", "visitor_destination_count_500m", "lodging_count_500m"],
    "urban_context": BASE + ["distance_to_amsterdam_center_m", "in_key_shopping_area"],
    "street_context": BASE + ["walkable_street_length_200m", "pedestrian_street_length_200m", "pedestrian_street_share_200m", "intersection_count_150m", "distance_to_nearest_intersection_m"],
    "poi_mix": BASE + ["active_poi_categories_300m", "retail_concentration_80_300", "horeca_concentration_120_300"],
    "high_street_and_key_area": BASE + ["high_street_shop_count_120m", "in_key_shopping_area"],
    "compact_market_context": BASE + ["high_street_shop_count_120m", "in_key_shopping_area", "lodging_count_500m", "horeca_count_200m"],
    "compact_market_context_300m": BASE + ["high_street_shop_count_120m", "in_key_shopping_area", "lodging_count_500m", "horeca_count_300m"],
    "curated_context": BASE + [
        "high_street_shop_count_120m",
        "nightlife_count_300m",
        "high_capacity_transit_count_500m",
        "visitor_destination_count_500m",
        "distance_to_amsterdam_center_m",
        "pedestrian_street_length_200m",
        "intersection_count_150m",
    ],
}

LOG_FEATURES = {
    name
    for names in GROUPS.values()
    for name in names
    if name not in {
        "in_official_shopping_area",
        "in_key_shopping_area",
        "pedestrian_street_share_200m",
        "retail_concentration_80_300",
        "horeca_concentration_120_300",
        "active_poi_categories_300m",
    }
}


def transformed_value(row: dict[str, Any], name: str) -> float:
    value = safe_float(row.get(name))
    return log1p(value) if name in LOG_FEATURES else value


def matrix(rows: list[dict[str, Any]], features: list[str]) -> list[list[float]]:
    return [[transformed_value(row, name) for name in features] for row in rows]


def evaluate_feature_set(rows: list[dict[str, Any]], name: str, features: list[str]) -> dict[str, Any]:
    x = matrix(rows, features)
    y = [log1p(safe_float(row["weekly_passers"])) for row in rows]
    best: dict[str, Any] | None = None
    for alpha in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
        predictions = leave_one_out_predictions("ridge", x, y, alpha, feature_names=features)
        result = metrics(y, predictions)
        result["rank_correlation"] = rank_correlation(y, predictions)
        if best is None or (result["rmse_log"], result["mae"]) < (best["rmse_log"], best["mae"]):
            best = {"feature_set": name, "alpha": alpha, "feature_count": len(features), **result}
    assert best is not None
    best["features"] = ";".join(features)
    return best


def nested_alpha_predictions(rows: list[dict[str, Any]], features: list[str]) -> tuple[list[float], list[float]]:
    x = matrix(rows, features)
    y = [log1p(safe_float(row["weekly_passers"])) for row in rows]
    predictions = []
    chosen_alphas = []
    for holdout in range(len(rows)):
        train_x = [row for idx, row in enumerate(x) if idx != holdout]
        train_y = [value for idx, value in enumerate(y) if idx != holdout]
        best_alpha = min(
            [0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
            key=lambda alpha: metrics(
                train_y,
                leave_one_out_predictions("ridge", train_x, train_y, alpha, feature_names=features),
            )["rmse_log"],
        )
        train_std, specs = standardize_fit(train_x, features)
        test_std = standardize_apply([x[holdout]], specs)
        predictions.append(predict_linear(test_std, fit_ridge(train_std, train_y, best_alpha))[0])
        chosen_alphas.append(best_alpha)
    return predictions, chosen_alphas


def main() -> None:
    rows = [row for row in read_csv(DATA_PROCESSED / "site_features_weekly.csv") if safe_float(row.get("weekly_passers")) > 0]
    results = [evaluate_feature_set(rows, name, features) for name, features in GROUPS.items()]

    for feature in sorted({feature for features in GROUPS.values() for feature in features} - set(BASE)):
        results.append(evaluate_feature_set(rows, f"baseline_plus__{feature}", BASE + [feature]))

    baseline = next(row for row in results if row["feature_set"] == "baseline")
    for row in results:
        row["rmse_improvement_vs_baseline_pct"] = round(
            100 * (baseline["rmse_log"] - row["rmse_log"]) / baseline["rmse_log"], 2
        )
    results.sort(key=lambda row: (row["rmse_log"], row["mae"]))
    write_csv(OUTPUT_TABLES / "feature_experiment_results.csv", results)

    y = [log1p(safe_float(row["weekly_passers"])) for row in rows]
    stability_rows = []
    for name in ["baseline", "compact_market_context"]:
        predictions, alphas = nested_alpha_predictions(rows, GROUPS[name])
        result = metrics(y, predictions)
        stability_rows.append(
            {
                "feature_set": name,
                "validation": "nested_loocv_alpha_selection",
                "rmse_log": round(result["rmse_log"], 4),
                "mae": round(result["mae"]),
                "mape_pct": round(result["mape_pct"], 1),
                "rank_correlation": round(rank_correlation(y, predictions), 3),
                "chosen_alphas": ";".join(str(alpha) for alpha in alphas),
            }
        )
    write_csv(OUTPUT_TABLES / "feature_experiment_stability.csv", stability_rows)
    print("Feature experiment complete")
    for row in results[:12]:
        print(
            f"{row['feature_set']:<58} alpha={row['alpha']:<5} "
            f"rmse_log={row['rmse_log']:.3f} mae={row['mae']:.0f} "
            f"improvement={row['rmse_improvement_vs_baseline_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
