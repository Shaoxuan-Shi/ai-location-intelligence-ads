# Model Results

Updated on 2026-09-18 from cached Amsterdam weekly Crowdmonitor history.

## Data Coverage

- Weekly history rows: 2,017
- Weekly dates: 65
- History range: 2023-01-02 to 2024-03-25
- Crowdmonitor locations in history: 34
- Usable model anchors with coordinates: 25 after merging one exactly co-located directional pair
- Official horeca points: 3,920
- OSM shop points in candidate bbox: 4,481
- OSM transit points in candidate bbox: 1,500
- OSM tourism/historic points in candidate bbox: 1,711

## Feature Distribution Check

| Feature | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| normal_week_median_passers | 3,415 | 160,390 | 463,537 |
| shop_count_80m | 0 | 13 | 28 |
| horeca_count_120m | 0 | 14 | 78 |
| transit_stop_count_250m | 0 | 6 | 57 |
| distance_to_nearest_transit_m | 9.2 | 151.6 | 345.3 |
| tourism_poi_count_300m | 1 | 38 | 90 |
| in_official_shopping_area | 0 | 0 | 1 |

The distribution check supports the move away from 1-5 manual scoring. Several variables vary meaningfully across the 25 locations, so percentile profiles and regression features are more informative than coarse bands.

## Model Comparison

Leave-one-out cross-validation:

| Model | Alpha | MAE | MAPE | RMSE log |
| --- | ---: | ---: | ---: | ---: |
| Huber-Ridge | 3.0 | 59,344 | 53.0% | 0.567 |
| Lasso | 0.03 | 64,586 | 52.4% | 0.576 |
| Ridge | 3.0 | 63,374 | 54.3% | 0.585 |
| mean baseline | 0.0 | 134,107 | 302.2% | 1.506 |

Huber-Ridge is the production model. It retains the same interpretable linear structure and Ridge regularization, while reducing the influence of a few anchors whose traffic is not explained by the current location features.

## Target and Robustness Experiment

The public sensor metadata confirms that `CS - ri. IJplein (Oost)` and `(West)` are two cameras at the same coordinates. Deleting either camera is arbitrary, while summing may double-count exposure without screen-orientation information. The production target therefore averages exactly co-located sensors week by week into one direction-neutral anchor.

The target pipeline also removes severe local sensor failures below 20% or above five times the raw site median and one citywide incomplete week. Ordinary holidays and seasonal changes remain in the data. The main comparison was:

| Target/model | RMSE log | MAPE | Rank correlation |
| --- | ---: | ---: | ---: |
| Original directional targets + Ridge | 0.590 | 55.1% | 0.858 |
| Direction-neutral targets + Ridge | 0.585 | 54.3% | 0.849 |
| Direction-neutral normal-week targets + Huber-Ridge | 0.567 | 53.0% | 0.874 |
| Plus short nearby roadwork filtering | 0.564 | 52.7% | 0.878 |

Roadwork filtering was rejected for production. Its incremental RMSE gain was only 0.003, while proximity to a registered work does not prove pedestrian disruption. The historical events API returned no records for the training window, so no event adjustment is claimed.

## Feature Experiment

The pipeline tests more than 30 individual additions and grouped feature sets using the same leave-one-anchor-out procedure. The selected compact market-context set adds:

- high-street retail count within 120m
- Amsterdam key-shopping-area membership
- lodging count within 500m
- horeca count within 200m

On the cleaned target, the compact ten-feature Ridge records RMSE log 0.585. Replacing horeca within 200m with 300m lowers Huber-Ridge RMSE to 0.562 but reduces rank correlation from 0.874 to 0.849. The 200m version remains in production because point accuracy and candidate ranking both matter.

Street-network proxies, intersection density, pedestrian-street share, transit hierarchy, and several multi-scale counts were tested but did not improve the held-out result. Screen orientation, visibility, and obstruction remain unavailable for arbitrary coordinates.

## Pedestrian-Network Experiment

A second experiment fetched 239,416 road, footpath, and crossing edges from four bounded regions of Amsterdam's official `loopfietsnetwerk` extract dated 2026-08-16. It tested:

- network length reachable within 400m walking distance
- sampled pedestrian route centrality within 800m
- transit-to-destination route centrality within 800m
- uninterrupted corridor continuity
- local mapped walkway width

The experiment first exposed and corrected an isolated-segment snapping failure at Molensteeg by requiring candidate points and destinations to join the main local walking-network component.

| Feature set | Nested LOOCV RMSE log | Nested MAE | Change in RMSE log |
| --- | ---: | ---: | ---: |
| Prior production features and directional target | 0.590 | 61,387 | baseline |
| + network reach | 0.597 | 58,854 | 1.1% worse |
| + corridor continuity | 0.598 | 62,207 | 1.3% worse |
| + pedestrian route centrality | 0.605 | 59,402 | 2.5% worse |
| + local walkway width | 0.613 | 48,299 | 3.8% worse |
| + all network features | 0.728 | 54,000 | 23.3% worse |

Walkway width reduced MAE but worsened RMSE log and percentage error, indicating uneven gains across high- and low-traffic anchors. None of the network additions improved the primary nested held-out metric, so no network feature was added to the production model. The result does not show that pedestrian routing is irrelevant; it shows that these public-data proxies were not sufficiently reliable with the available spatial anchors. This experiment predates the direction-neutral target and is not used to claim a new production gain.

## Current Main Model

Selected model:

```text
Huber-Ridge regression on log1p(direction-neutral normal-week median), alpha = 3.0, delta = 1.35
```

Validation summary:

- train RMSE log: 0.443
- leave-one-anchor-out RMSE log: 0.567
- leave-one-anchor-out MAPE: 53.0%
- median held-out percentage error: 33.2%
- mean-baseline leave-one-anchor-out RMSE log: 1.506
- leave-one-anchor-out rank correlation: 0.874

Largest positive standardized coefficients:

- `lodging_count_500m`
- `high_street_shop_count_120m`
- `tourism_poi_count_300m`
- `transit_stop_count_250m`

Largest negative standardized coefficient:

- `distance_to_nearest_transit_m`

These coefficients are associations, not causal effects. With only 25 anchors and correlated place features, they should be used to explain the model at a high level rather than interpreted in isolation.

## Important Diagnostic Note

The model still underpredicts some high-performing city-center locations, especially Damstraat. The dashboard exposes the local held-out errors and feature coverage rather than hiding the issue. This suggests that the current feature set still misses some local pedestrian-flow mechanisms, such as:

- exact street-network centrality
- pedestrian corridor continuity
- intersection visibility
- nearby landmarks not fully captured as POI counts
- screen-facing or street-facing exposure quality

This limitation should be actively discussed in the portfolio narrative. It strengthens the project if framed as a validation insight rather than hidden as a model failure.

## How to Use the Outputs

- Use `predicted_weekly_passers` for model-based typical weekly exposure.
- Use percentile profiles for explanation and stakeholder discussion.
- Use scenario ranking only as a campaign-fit view, not as a footfall prediction.
- Use prediction intervals and reliability evidence to decide which sites need field validation.
- Use the custom evaluator for new candidate coordinates.
