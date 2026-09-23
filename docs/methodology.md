# Methodology

## Modeling Philosophy

The project separates four ideas that are often mixed together:

1. **Foot-traffic prediction**: estimate weekly passers using historical anchor data.
2. **Relative explanation**: show where each location sits in the distribution of retail, horeca, transit, and tourism features.
3. **Business preference ranking**: optionally adjust recommendations for a campaign objective.
4. **Uncertainty**: communicate when the model should be trusted less.

This separation avoids treating subjective business weights as if they were scientific footfall estimates.

## Why Weekly Passers

The current model predicts typical weekly passers because the available Crowdmonitor data provides weekly anchors, and weekly exposure is more stable than daily or hourly traffic for early-stage outdoor advertising site selection.

Daily and hourly profiles can be useful later, but they would add scope and noise before the core ranking problem is solved.

The current pipeline audits weekly history and uses a normal-week median as the main target. Co-located directional sensors are averaged week by week because an arbitrary coordinate does not provide screen orientation. Weeks below 20% or above five times a site's raw median, plus a clearly incomplete citywide week, are excluded before calculating the median. This removes severe sensor failures without treating ordinary seasonal variation as noise.

The event API returned no records for the 2023-2024 training window. Historical WIOR roadworks were tested separately, but excluding short nearby works improved RMSE log by only 0.003 and removed many weeks whose pedestrian impact was uncertain. Roadwork filtering is documented as an experiment, not used in production.

## Features and Factors

The original assignment discussed high-level factors such as retail context, horeca intensity, and transit accessibility. In this project, each factor is translated into computable features.

| Conceptual factor | Computable feature examples |
| --- | --- |
| Retail density | `shop_count_80m`, `shop_count_120m` |
| Horeca intensity | `horeca_count_120m` |
| Transit accessibility | `transit_stop_count_250m`, `distance_to_nearest_transit_m` |
| Tourism attractiveness | `tourism_poi_count_300m` |
| Retail context | `in_official_shopping_area`, `shopping_area_category` |

An expanded feature experiment also tests multiple radii, high-street retail types, lodging, nightlife, visitor destinations, city-centre distance, transit hierarchy, walkable-street length, pedestrian-street share, and intersection density. Every candidate feature must be computable for both training anchors and a new map-selected location.

A separate bounded experiment uses Amsterdam's official `loopfietsnetwerk` to test pedestrian-network reach, sampled route centrality, destination-weighted route centrality, corridor continuity, and local walkway width. None reduced nested leave-one-anchor-out RMSE log, so they are documented as rejected experimental features rather than added to the production model.

Percentile versions of these features are generated for interpretation. The model can still use raw or transformed numeric features.

## Modeling Plan

The main target is:

```text
direction_neutral_normal_week_median
```

The preferred target transformation is:

```text
log1p(direction_neutral_normal_week_median)
```

This reduces the influence of very large city-center counts and makes simple regression more stable.

Models:

- ordinary log-linear regression as an interpretable baseline
- Huber-Ridge regression as the main robust regularized model
- lasso-style feature selection if the data supports it
- a simple decision-tree benchmark if the sample size supports it

The final recommendation does not depend on one fragile model. Outputs include held-out error, rank stability, and reproducible feature and target experiments. The selected market-context set adds high-street retail within 120m, core shopping-area membership, lodging within 500m, and horeca within 200m. On the cleaned 25-anchor target, Huber-Ridge reaches LOOCV RMSE log 0.567 versus 0.585 for ordinary Ridge, with rank correlation 0.874.

## Overfitting Controls

- Keep the feature set small and interpretable.
- Use regularization for the main model.
- Compare against simple baselines.
- Use leave-one-out or k-fold cross-validation depending on sample size.
- Avoid over-interpreting individual coefficients when anchors are limited.
- Report uncertainty and limitations explicitly.

## Custom Candidate Evaluation

The dashboard and CLI can evaluate a new Amsterdam candidate point by latitude and longitude. The product calculates the same feature set around the point, applies the trained model, and returns:

- predicted typical weekly passers
- prediction interval
- reliability evidence
- percentile profile
- nearest comparable anchor locations

Map-click evaluation uses cached Amsterdam POI and shopping-area data, so it does not require a local server or live API call for each click.

## Reliability Evidence

The dashboard intentionally avoids a single opaque confidence score. Instead, it shows the evidence that affects trust:

- comparable-anchor support
- similar-anchor validation error
- historical stability from weekly IQR/median
- training-range check with both single-feature extrapolation and joint feature similarity

The prediction interval remains beside the point estimate as the primary uncertainty range. The evidence cards explain whether weak comparables, volatile historical traffic, or an unusual feature profile make transfer to the candidate less defensible.
