from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from utils import DATA_PROCESSED, OUTPUT_TABLES, expm1, haversine_m, log1p, percentile_ranks, quantile, read_csv, safe_float, write_csv, write_json


RAW_FEATURES = [
    "shop_count_80m",
    "horeca_count_120m",
    "transit_stop_count_250m",
    "distance_to_nearest_transit_m",
    "tourism_poi_count_300m",
    "in_official_shopping_area",
    "high_street_shop_count_120m",
    "in_key_shopping_area",
    "lodging_count_500m",
    "horeca_count_200m",
]

IDENTITY_FEATURES = {"in_official_shopping_area", "in_key_shopping_area"}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*matrix)]


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def gaussian_solve(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            aug[pivot][col] = 1e-12
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


@dataclass
class FeatureSpec:
    name: str
    mean: float
    std: float


def transform_row(row: dict[str, Any]) -> list[float]:
    return [
        safe_float(row[name]) if name in IDENTITY_FEATURES else log1p(safe_float(row[name]))
        for name in RAW_FEATURES
    ]


def add_nearby_anchor_feature(rows: list[dict[str, Any]]) -> None:
    global_median = sorted(safe_float(row["weekly_passers"]) for row in rows)[len(rows) // 2]
    for row in rows:
        lat = safe_float(row["lat"])
        lon = safe_float(row["lon"])
        weighted_sum = 0.0
        weight_total = 0.0
        nearest_distance = 9999.0
        nearest_name = ""
        for other in rows:
            if other is row or other.get("location_name") == row.get("location_name"):
                continue
            distance = haversine_m(lat, lon, safe_float(other["lat"]), safe_float(other["lon"]))
            nearest_distance = min(nearest_distance, distance)
            if nearest_distance == distance:
                nearest_name = other.get("location_name", "")
            weight = 1.0 / max(distance, 80.0)
            weighted_sum += safe_float(other["weekly_passers"]) * weight
            weight_total += weight
        row["nearby_anchor_median_passers"] = round(weighted_sum / weight_total if weight_total else global_median)
        row["nearest_anchor_distance_m"] = round(nearest_distance, 1)
        row["nearest_anchor_name"] = nearest_name


def standardize_fit(x: list[list[float]], feature_names: list[str]) -> tuple[list[list[float]], list[FeatureSpec]]:
    specs = []
    columns = transpose(x)
    for name, col in zip(feature_names, columns):
        mu = mean(col)
        variance = mean([(value - mu) ** 2 for value in col])
        std = math.sqrt(variance) or 1.0
        specs.append(FeatureSpec(name=name, mean=mu, std=std))
    return standardize_apply(x, specs), specs


def standardize_apply(x: list[list[float]], specs: list[FeatureSpec]) -> list[list[float]]:
    return [[(value - spec.mean) / spec.std for value, spec in zip(row, specs)] for row in x]


def add_intercept(x: list[list[float]]) -> list[list[float]]:
    return [[1.0] + row for row in x]


def fit_ridge(x: list[list[float]], y: list[float], alpha: float) -> list[float]:
    x_i = add_intercept(x)
    xt = transpose(x_i)
    xtx = [[dot(row, col) for col in xt] for row in xt]
    for idx in range(1, len(xtx)):
        xtx[idx][idx] += alpha
    xty = [dot(row, y) for row in xt]
    return gaussian_solve(xtx, xty)


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
    x: list[list[float]], y: list[float], alpha: float, delta: float = 1.35, iterations: int = 30
) -> list[float]:
    coefficients = fit_ridge(x, y, alpha)
    for _ in range(iterations):
        residuals = [actual - predicted for actual, predicted in zip(y, predict_linear(x, coefficients))]
        residual_median = quantile(residuals, 0.5)
        scale = quantile([abs(value - residual_median) for value in residuals], 0.5) / 0.6745
        scale = max(scale, 1e-6)
        threshold = delta * scale
        weights = [1.0 if abs(value) <= threshold else threshold / abs(value) for value in residuals]
        updated = fit_weighted_ridge(x, y, alpha, weights)
        if max(abs(a - b) for a, b in zip(coefficients, updated)) < 1e-6:
            return updated
        coefficients = updated
    return coefficients


def predict_linear(x: list[list[float]], coefs: list[float]) -> list[float]:
    return [coefs[0] + dot(row, coefs[1:]) for row in x]


def soft_threshold(value: float, alpha: float) -> float:
    if value > alpha:
        return value - alpha
    if value < -alpha:
        return value + alpha
    return 0.0


def fit_lasso_cd(x: list[list[float]], y: list[float], alpha: float, iterations: int = 1000) -> list[float]:
    # Coordinate descent on standardized features; intercept is the mean target.
    n = len(y)
    p = len(x[0]) if x else 0
    intercept = mean(y)
    centered_y = [value - intercept for value in y]
    beta = [0.0] * p
    predictions = [0.0] * n
    for _ in range(iterations):
        max_change = 0.0
        for j in range(p):
            residual = [centered_y[i] - predictions[i] + x[i][j] * beta[j] for i in range(n)]
            rho = sum(x[i][j] * residual[i] for i in range(n)) / n
            denom = sum(x[i][j] ** 2 for i in range(n)) / n or 1.0
            updated = soft_threshold(rho, alpha) / denom
            change = updated - beta[j]
            if change:
                for i in range(n):
                    predictions[i] += x[i][j] * change
            beta[j] = updated
            max_change = max(max_change, abs(change))
        if max_change < 1e-8:
            break
    return [intercept] + beta


def fit_tree_stump(x: list[list[float]], y: list[float], min_leaf: int = 3) -> dict[str, Any]:
    best = {"feature": None, "threshold": None, "left": mean(y), "right": mean(y), "sse": float("inf")}
    n = len(y)
    if n < 2 * min_leaf:
        return best
    for feature_idx in range(len(x[0])):
        values = sorted(set(row[feature_idx] for row in x))
        thresholds = [(a + b) / 2 for a, b in zip(values, values[1:])]
        for threshold in thresholds:
            left_y = [target for row, target in zip(x, y) if row[feature_idx] <= threshold]
            right_y = [target for row, target in zip(x, y) if row[feature_idx] > threshold]
            if len(left_y) < min_leaf or len(right_y) < min_leaf:
                continue
            left_mean = mean(left_y)
            right_mean = mean(right_y)
            sse = sum((value - left_mean) ** 2 for value in left_y) + sum((value - right_mean) ** 2 for value in right_y)
            if sse < best["sse"]:
                best = {
                    "feature": feature_idx,
                    "threshold": threshold,
                    "left": left_mean,
                    "right": right_mean,
                    "sse": sse,
                }
    return best


def predict_tree_stump(x: list[list[float]], model: dict[str, Any]) -> list[float]:
    feature = model.get("feature")
    if feature is None:
        return [model["left"] for _ in x]
    threshold = model["threshold"]
    return [model["left"] if row[feature] <= threshold else model["right"] for row in x]


def metrics(y_true_log: list[float], y_pred_log: list[float]) -> dict[str, float]:
    pred = [expm1(value) for value in y_pred_log]
    true = [expm1(value) for value in y_true_log]
    mae = mean([abs(a - b) for a, b in zip(true, pred)])
    mape = mean([abs(a - b) / max(a, 1.0) for a, b in zip(true, pred)]) * 100
    rmse_log = math.sqrt(mean([(a - b) ** 2 for a, b in zip(y_true_log, y_pred_log)]))
    return {"mae": mae, "mape_pct": mape, "rmse_log": rmse_log}


def rank_correlation(y_true: list[float], y_pred: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        ordered = sorted((value, idx) for idx, value in enumerate(values))
        output = [0.0] * len(values)
        for rank, (_, idx) in enumerate(ordered, start=1):
            output[idx] = float(rank)
        return output

    if len(y_true) < 2:
        return 0.0
    a = ranks(y_true)
    b = ranks(y_pred)
    am = mean(a)
    bm = mean(b)
    num = sum((x - am) * (y - bm) for x, y in zip(a, b))
    den_a = math.sqrt(sum((x - am) ** 2 for x in a))
    den_b = math.sqrt(sum((y - bm) ** 2 for y in b))
    return num / (den_a * den_b) if den_a and den_b else 0.0


def leave_one_out_predictions(
    model_name: str,
    x: list[list[float]],
    y: list[float],
    alpha: float = 0.0,
    *,
    feature_names: list[str] | None = None,
    huber_delta: float = 1.35,
) -> list[float]:
    preds = []
    for holdout in range(len(y)):
        train_x = [row for i, row in enumerate(x) if i != holdout]
        train_y = [value for i, value in enumerate(y) if i != holdout]
        test_x = [x[holdout]]
        names = feature_names or RAW_FEATURES
        train_x_std, specs = standardize_fit(train_x, names)
        test_x_std = standardize_apply(test_x, specs)
        if model_name == "mean_baseline":
            pred = [mean(train_y)]
        elif model_name == "ridge":
            coefs = fit_ridge(train_x_std, train_y, alpha)
            pred = predict_linear(test_x_std, coefs)
        elif model_name == "huber_ridge":
            coefs = fit_huber_ridge(train_x_std, train_y, alpha, huber_delta)
            pred = predict_linear(test_x_std, coefs)
        elif model_name == "lasso":
            coefs = fit_lasso_cd(train_x_std, train_y, alpha)
            pred = predict_linear(test_x_std, coefs)
        elif model_name == "tree_stump":
            tree = fit_tree_stump(train_x_std, train_y, min_leaf=max(3, len(train_y) // 8))
            pred = predict_tree_stump(test_x_std, tree)
        else:
            raise ValueError(model_name)
        preds.append(pred[0])
    return preds


def choose_ridge_alpha(x: list[list[float]], y: list[float]) -> tuple[float, list[dict[str, float]]]:
    rows = []
    best_alpha = 0.0
    best_rmse = float("inf")
    for alpha in [0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]:
        preds = leave_one_out_predictions("ridge", x, y, alpha)
        row = {"model": "ols" if alpha == 0 else "ridge", "alpha": alpha, **metrics(y, preds)}
        rows.append(row)
        if row["rmse_log"] < best_rmse:
            best_rmse = row["rmse_log"]
            best_alpha = alpha
    return best_alpha, rows


def lasso_benchmark(x: list[list[float]], y: list[float]) -> list[dict[str, float]]:
    rows = []
    for alpha in [0.01, 0.03, 0.1, 0.3]:
        preds = leave_one_out_predictions("lasso", x, y, alpha)
        rows.append({"model": "lasso_cd", "alpha": alpha, **metrics(y, preds)})
    return rows


def main() -> None:
    rows = read_csv(DATA_PROCESSED / "site_features_weekly.csv")
    rows = [row for row in rows if safe_float(row.get("weekly_passers")) > 0]
    if len(rows) < 8:
        raise RuntimeError("Need at least 8 usable anchors for regression.")
    add_nearby_anchor_feature(rows)
    nearby_percentiles = percentile_ranks(
        [safe_float(row.get("nearby_anchor_median_passers")) for row in rows], higher_is_better=True
    )
    for row, percentile in zip(rows, nearby_percentiles):
        row["nearby_anchor_median_passers_percentile"] = percentile

    x_raw = [transform_row(row) for row in rows]
    y = [log1p(safe_float(row["weekly_passers"])) for row in rows]

    best_alpha, metric_rows = choose_ridge_alpha(x_raw, y)
    huber_delta = 1.35
    selected_loo_log = leave_one_out_predictions(
        "huber_ridge", x_raw, y, best_alpha, huber_delta=huber_delta
    )
    metric_rows.append(
        {
            "model": "huber_ridge",
            "alpha": best_alpha,
            "huber_delta": huber_delta,
            **metrics(y, selected_loo_log),
        }
    )
    baseline_preds = leave_one_out_predictions("mean_baseline", x_raw, y)
    metric_rows.append({"model": "mean_baseline", "alpha": 0.0, **metrics(y, baseline_preds)})
    if len(rows) >= 15:
        metric_rows.extend(lasso_benchmark(x_raw, y))
        tree_preds = leave_one_out_predictions("tree_stump", x_raw, y)
        metric_rows.append({"model": "tree_stump", "alpha": 0.0, **metrics(y, tree_preds)})
    x_std, specs = standardize_fit(x_raw, RAW_FEATURES)
    final_coefs = fit_huber_ridge(x_std, y, best_alpha, huber_delta)
    fitted_log = predict_linear(x_std, final_coefs)
    residuals = [actual - pred for actual, pred in zip(y, fitted_log)]
    rmse_log = math.sqrt(mean([resid**2 for resid in residuals]))
    selected_loo_metrics = metrics(y, selected_loo_log)
    fit_metrics = metrics(y, fitted_log)
    baseline_metrics = metrics(y, baseline_preds)
    ridge_benchmark_log = leave_one_out_predictions("ridge", x_raw, y, best_alpha)
    ridge_benchmark_metrics = metrics(y, ridge_benchmark_log)
    absolute_percentage_errors = sorted(
        abs(expm1(actual) - expm1(predicted)) / max(expm1(actual), 1.0)
        for actual, predicted in zip(y, selected_loo_log)
    )
    median_absolute_percentage_error = quantile(absolute_percentage_errors, 0.5) * 100
    selected_rank_corr = rank_correlation([expm1(value) for value in y], [expm1(value) for value in selected_loo_log])
    interval_log_margin = quantile([abs(actual - predicted) for actual, predicted in zip(y, selected_loo_log)], 0.8)

    prediction_rows: list[dict[str, Any]] = []
    for row, pred_log, resid, loo_pred_log, x_row_std in zip(rows, fitted_log, residuals, selected_loo_log, x_std):
        low = expm1(pred_log - interval_log_margin)
        high = expm1(pred_log + interval_log_margin)
        point = expm1(pred_log)
        width_ratio = (high - low) / max(point, 1.0)
        actual = safe_float(row["weekly_passers"])
        loo_pred = expm1(loo_pred_log)
        cv_abs_error = abs(actual - loo_pred)
        cv_abs_pct_error = cv_abs_error / max(actual, 1.0)
        mean_abs_z = mean([abs(value) for value in x_row_std])
        max_abs_z = max(abs(value) for value in x_row_std) if x_row_std else 0.0
        prediction_rows.append(
            {
                **row,
                "predicted_weekly_passers": round(point),
                "prediction_low": round(low),
                "prediction_high": round(high),
                "actual_minus_predicted": round(safe_float(row["weekly_passers"]) - point),
                "absolute_error": round(abs(safe_float(row["weekly_passers"]) - point)),
                "loocv_predicted_weekly_passers": round(loo_pred),
                "loocv_absolute_error": round(cv_abs_error),
                "loocv_absolute_percentage_error": round(cv_abs_pct_error * 100, 1),
                "interval_width_ratio": round(width_ratio, 3),
                "feature_mean_abs_z": round(mean_abs_z, 3),
                "feature_max_abs_z": round(max_abs_z, 3),
                "model": "huber_ridge_log_linear",
            }
        )

    prediction_rows.sort(key=lambda row: row["predicted_weekly_passers"], reverse=True)
    for idx, row in enumerate(prediction_rows, start=1):
        row["predicted_rank"] = idx

    coef_rows = [{"term": "intercept", "coefficient_log_scale": final_coefs[0]}]
    for spec, coef in zip(specs, final_coefs[1:]):
        coef_rows.append(
            {
                "term": spec.name,
                "coefficient_log_scale": round(coef, 4),
                "feature_mean_transformed": round(spec.mean, 4),
                "feature_std_transformed": round(spec.std, 4),
            }
        )

    metric_rows.sort(key=lambda row: (row["rmse_log"], row["mae"]))
    residual_rows = sorted(
        [
            {
                "location_name": row["location_name"],
                "actual_median_weekly_passers": row["weekly_passers"],
                "loocv_predicted_weekly_passers": row["loocv_predicted_weekly_passers"],
                "loocv_absolute_percentage_error": row["loocv_absolute_percentage_error"],
                "actual_minus_loocv_predicted": round(
                    safe_float(row["weekly_passers"]) - safe_float(row["loocv_predicted_weekly_passers"])
                ),
            }
            for row in prediction_rows
        ],
        key=lambda row: safe_float(row["loocv_absolute_percentage_error"]),
        reverse=True,
    )
    validation_summary = [
        {
            "metric": "selected_model",
            "value": "huber_ridge_log_linear",
            "detail": f"alpha={best_alpha}, Huber delta={huber_delta}; robust to a few atypical anchors",
        },
        {
            "metric": "train_rmse_log",
            "value": round(fit_metrics["rmse_log"], 3),
            "detail": f"Fit on all {len(rows)} direction-neutral normal-week anchor medians",
        },
        {
            "metric": "loocv_rmse_log",
            "value": round(selected_loo_metrics["rmse_log"], 3),
            "detail": "Leave-one-anchor-out validation",
        },
        {
            "metric": "mean_baseline_loocv_rmse_log",
            "value": round(baseline_metrics["rmse_log"], 3),
            "detail": "Mean-only baseline",
        },
        {
            "metric": "ridge_benchmark_rmse_log",
            "value": round(ridge_benchmark_metrics["rmse_log"], 3),
            "detail": "Same target and features without Huber's robust residual weighting",
        },
        {
            "metric": "loocv_mae_passers",
            "value": round(selected_loo_metrics["mae"]),
            "detail": "Average absolute error in weekly passers",
        },
        {
            "metric": "loocv_mape",
            "value": f"{selected_loo_metrics['mape_pct']:.1f}%",
            "detail": "Average absolute percentage error; sensitive to low-traffic anchors",
        },
        {
            "metric": "median_held_out_percentage_error",
            "value": f"{median_absolute_percentage_error:.1f}%",
            "detail": "A typical anchor's absolute percentage miss; less dominated by Zeeburg-like low-volume cases",
        },
        {
            "metric": "loocv_rank_correlation",
            "value": round(selected_rank_corr, 3),
            "detail": "Whether the model preserves relative site ranking",
        },
    ]
    write_csv(OUTPUT_TABLES / "model_metrics.csv", metric_rows)
    write_csv(OUTPUT_TABLES / "model_validation_summary.csv", validation_summary)
    write_csv(OUTPUT_TABLES / "largest_residuals.csv", residual_rows[:8])
    write_csv(OUTPUT_TABLES / "feature_coefficients.csv", coef_rows)
    write_csv(OUTPUT_TABLES / "site_predictions_weekly.csv", prediction_rows)
    write_json(
        OUTPUT_TABLES / "model_artifact.json",
        {
            "model": "huber_ridge_log_linear",
            "target": rows[0].get("target_definition", "weekly_passers") if rows else "weekly_passers",
            "target_column": "weekly_passers",
            "target_transform": "log1p",
            "alpha": best_alpha,
            "huber_delta": huber_delta,
            "rmse_log_fit": rmse_log,
            "interval_log_margin": interval_log_margin,
            "interval_method": "80th percentile absolute LOOCV log residual",
            "coefficients": final_coefs,
            "features": [
                {
                    "name": spec.name,
                    "mean": spec.mean,
                    "std": spec.std,
                    "transform": "identity" if spec.name in IDENTITY_FEATURES else "log1p",
                }
                for spec in specs
            ],
            "training_rows": [
                {
                    "location_name": row["location_name"],
                    "lat": safe_float(row["lat"]),
                    "lon": safe_float(row["lon"]),
                    "weekly_passers": safe_float(row["weekly_passers"]),
                    "variability_ratio": safe_float(row.get("variability_ratio")),
                    "nearest_anchor_distance_m": safe_float(row.get("nearest_anchor_distance_m")),
                    "nearest_anchor_name": row.get("nearest_anchor_name", ""),
                    "raw_features": {
                        **{name: safe_float(row[name]) for name in RAW_FEATURES},
                        "nearby_anchor_median_passers": safe_float(row.get("nearby_anchor_median_passers")),
                    },
                }
                for row in rows
            ],
        },
    )
    print(
        f"Modeled {len(rows)} weekly anchors; selected Huber-Ridge "
        f"alpha={best_alpha}, delta={huber_delta}"
    )


if __name__ == "__main__":
    main()
