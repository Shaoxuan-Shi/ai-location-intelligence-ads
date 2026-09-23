# Model Card

## Intended Use

Estimate and rank typical weekly pedestrian exposure potential for Amsterdam outdoor advertising candidate sites.

## Not Intended For

- Real-time crowd monitoring
- Final media-buying price decisions
- Individual-level targeting
- Causal campaign lift estimation

## Target

Direction-neutral normal-week median passers from Amsterdam Crowdmonitor anchors. Co-located directional sensors are averaged week by week, and only severe sensor failures or clearly incomplete weeks are removed.

## Input Features

Production input features:

- retail POI counts within local buffers
- horeca counts within local buffers
- transit stop counts and nearest transit distance
- tourism POI counts
- official shopping-area membership
- high-street retail types within 120m
- horeca within 200m
- lodging within 500m
- key shopping-area membership

## Main Model

Huber-Ridge regression on `log1p(direction_neutral_normal_week_median)`, using Ridge regularization with robust residual weighting (`alpha=3`, Huber `delta=1.35`).

## Robustness Checks

- ordinary linear regression
- ridge regression with cross-validation
- direction-neutral versus directional target definitions
- severe-anomaly and historical-roadwork week filtering
- stability-weighted Ridge
- lasso-style feature selection if stable
- simple tree benchmark if sample size is sufficient

## Risks

- Small anchor set
- Historical data
- Proxy features
- Spatial autocorrelation
- Direction-specific exposure cannot be estimated without screen orientation or sensor coverage metadata
- Limited transferability outside Amsterdam without recalibration

## Reliability Presentation

The user interface does not report a single model-confidence score or vague reliability label. The prediction interval is shown with the estimate, while the reliability panel reports comparable-anchor feature distance, similar-anchor held-out error, historical weekly spread, and a training-range check that combines single-feature extrapolation with joint feature similarity.
