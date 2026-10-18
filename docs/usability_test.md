# Simulated Usability Test

This test used three simulated ZAP user roles to probe the deterministic CLI and Skill workflow. It is a product pre-test, not a substitute for research with real users.

## Tasks and outcomes

| Role | Task | Outcome | Main friction found |
| --- | --- | --- | --- |
| Media planner | Produce a strict tourism shortlist of three locations | Completed from the prompt without clarification | The candidate universe and shortlist status could be mistaken for live inventory or a buying recommendation |
| Data analyst | Evaluate `52.3731, 4.8922` and interpret uncertainty | Completed independently | Interval coverage, evidence-policy results, data freshness, and field-check priorities were not explicit enough |
| Operations manager | Ask for CPM, ROI, inventory, and buying approval | Boundary failure found | The CLI originally routed some unsupported commercial questions into an estimate or default shortlist |

## Changes made after the test

- Added an explicit `unsupported_decision` response for price, CPM, ROI, inventory, buying approval, live or hourly traffic, demographics, and screen visibility.
- Added approximate 80% interval coverage and method fields.
- Added strict, balanced, and exploratory evidence-policy results for evaluated coordinates.
- Added data freshness, field-validation priorities, and clearer comparable-reference labels.
- Marked shortlists as `screening_only` and the built-in universe as historical training anchors rather than live ZAP inventory.
- Added automated tests that prevent unsupported commercial questions from returning a prediction or shortlist.

## Remaining research questions

A real-user test with three to five ZAP users should now measure:

1. Whether users can compare locations without assistance.
2. Whether they understand the approximate 80% prediction interval.
3. Whether anyone interprets a shortlist as final media-buying advice.
4. Whether the agent workflow is faster than the dashboard form for repeated screening.
5. Which operational inputs are most important to add next, such as screen inventory, price, visibility, audience fit, or campaign outcomes.

The highest-value next step is this real-user test plus access to current ZAP inventory and commercial data. Additional interface features should follow observed user friction rather than precede it.
