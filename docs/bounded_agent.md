# Bounded Agent Architecture

## Purpose

The V2 agent supports one bounded job: turn a saved set of Amsterdam outdoor-advertising candidate locations into an evidence-based shortlist.

It is not a general chatbot. It cannot modify model predictions, access unregistered data, or make final media-buying decisions.

## Workflow

1. Evaluate a candidate from coordinates or a map click.
2. Save candidates to the browser workspace.
3. Read a campaign brief from the controls or a natural-language request.
4. Apply transparent campaign-context and evidence filters.
5. Rank eligible candidates by predicted typical weekly passers.
6. Return a ranked shortlist and explicit exclusion reasons.

## Registered Tools

- `save_current_candidate`
- `load_sample_candidates`
- `clear_candidate_workspace`
- `compare_saved_candidates`
- `set_campaign_brief`
- `build_shortlist`
- `explain_candidate_status`
- `explain_exclusions`
- `explain_current_location`

The deterministic planner maps a user request to one or more of these tools. A request such as "Build a strict tourism shortlist of three sites" updates the brief, applies the filters, ranks eligible candidates, and returns the shortlist in sequence.

Questions can refer to a saved candidate by name. If the name is missing or ambiguous, the agent returns a boundary message instead of silently answering for another location. Caution-signal questions use a separate intent from model-driver questions.

## Filter First, Rank Second

Campaign preferences do not change the footfall model or its coefficients.

- Maximum exposure applies no contextual filter.
- Tourism, retail, and transit objectives require the corresponding context percentile to be at least 60.
- Strict evidence requires zero out-of-range inputs, comparable-anchor error no greater than 75%, historical spread no greater than 45%, joint similarity of at least 25%, and at least one typical feature match.
- Balanced evidence allows at most one out-of-range input, comparable-anchor error no greater than 150%, and joint similarity of at least 10%.
- Exploratory evidence does not exclude candidates; caution signals remain visible.

Eligible candidates are then ranked only by the existing predicted typical weekly passers. This keeps business preference separate from traffic estimation and avoids an opaque weighted score.

## State And Auditability

The browser stores up to 12 candidate evaluations locally. The user interface shows only decision-relevant predictions, campaign fit, reliability evidence, status, and exclusion reasons. Internal execution steps are available to tests but are not presented as a user-facing feature.

## Data Boundaries

The agent cannot determine:

- media price or CPM
- inventory availability
- audience demographics
- screen orientation or visibility
- campaign ROI
- hourly, daily, or live traffic

Those outputs require additional data sources and cannot be inferred from the current weekly Crowdmonitor model.

## Future Language-Model Planner

A language model may later replace the deterministic intent parser for more flexible phrasing and clarification questions. It should receive only the registered tool schemas and structured tool outputs. All numerical claims must remain grounded in deterministic tools, with user confirmation before any external or irreversible action.
