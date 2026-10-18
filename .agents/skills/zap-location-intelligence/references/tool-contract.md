# Deterministic tool contract

## Runtime source

`outputs/runtime/runtime_bundle.json` contains the compact POIs, shopping areas, anchor evidence, model coefficients, thresholds, schema version, model version, and data-as-of date required for offline evaluation. It is generated from the same context used by the static Dashboard.

Use `src/runtime_engine.py` for location evaluation and `src/shortlist.py` for campaign filtering and ranking. `src/agent_cli.py` is the stable command-line adapter for skills and other agents.

## Supported requests

### Evaluate a coordinate

```bash
python3 src/agent_cli.py --json "Estimate 52.3731 4.8922"
```

The response has `kind=location_evaluation` and a `candidate` object containing:

- `predicted_weekly_passers`
- `prediction_low` and `prediction_high`
- `interval_coverage`, `interval_method`, and `interval_width_ratio`
- retail, horeca, transit, and tourism percentiles
- similar-anchor error and historical spread
- training-range and feature-support evidence
- comparable anchors
- runtime version and data date
- per-policy pass/fail reasons, freshness warning, and field-validation priorities

### Build a shortlist

```bash
python3 src/agent_cli.py --json "Build a strict tourism top 3 shortlist"
```

The response has `kind=campaign_shortlist`, `decision_status=screening_only`, a normalized `brief`, ordered `field_validation_candidates`, candidate-universe metadata, counts, and explicit exclusion reasons. The current built-in universe is historical training anchors, not live ZAP inventory.

### Handle unsupported decisions

CPM, ROI, price, inventory, buying approval, live/hourly traffic, demographics, and screen-visibility requests return `kind=unsupported_decision`. Report `unsupported_fields`, `missing_inputs`, and `allowed_next_steps`. Do not add a prediction or shortlist to that response.

## Selection rules

- `exposure`: no contextual eligibility filter.
- `tourism`, `retail`, or `transit`: matching context percentile must be at least 60.
- `strict`: zero out-of-range inputs, similar-anchor error at most 75%, historical spread at most 45%, joint similarity at least 25%, and at least one typical feature match.
- `balanced`: at most one out-of-range input, similar-anchor error at most 150%, and joint similarity at least 10%.
- `exploratory`: do not exclude on evidence; keep cautions visible.
- Rank eligible candidates only by predicted typical weekly passers.

## Unsupported decisions

Return the limitation and the missing input when asked for media price, CPM, inventory availability, audience demographics, screen direction or visibility, campaign ROI, hourly traffic, or live traffic. Recommend field validation or an appropriate operational data source; do not calculate a substitute.
