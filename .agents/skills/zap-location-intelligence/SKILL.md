---
name: zap-location-intelligence
description: Evaluate Amsterdam outdoor-advertising candidate coordinates, explain weekly pedestrian-exposure estimates and reliability evidence, compare known sites, and build campaign shortlists for exposure, tourism, retail, or transit objectives. Use for ZAP Operations location screening, Amsterdam footfall questions, campaign shortlist requests, comparable-anchor explanations, and field-validation prioritization in this repository.
---

# ZAP Location Intelligence

Use the repository's versioned deterministic runtime. Do not reproduce the model, invent a weighted score, or estimate values in prose.

## Run the workflow

1. Resolve the project root containing both `src/agent_cli.py` and `outputs/runtime/runtime_bundle.json`.
2. Convert the request into one of these supported jobs:
   - evaluate an Amsterdam latitude/longitude;
   - explain a named training-anchor location;
   - build a shortlist for `exposure`, `tourism`, `retail`, or `transit`;
   - apply `strict`, `balanced`, or `exploratory` evidence policy.
3. Run from the project root:

   ```bash
   python3 src/agent_cli.py --json "<user request>"
   ```

4. If `kind=unsupported_decision`, return the refusal, missing inputs, and allowed next steps without adding an estimate or shortlist.
5. Treat other returned JSON as the only source for numerical claims and shortlist membership.
6. State the estimate, interval coverage, evidence cautions, data-as-of date, candidate-universe type, and practical next action.

Read [references/tool-contract.md](references/tool-contract.md) when integrating the runtime, interpreting fields, or handling unsupported requests.

## Preserve decision boundaries

- Describe outputs as early-stage Amsterdam screening estimates, never observed live traffic.
- Interpret `Shortlisted` as priority for field validation, not approval to buy media.
- Keep campaign context as an eligibility filter. Rank eligible candidates only by predicted typical weekly passers.
- Do not infer price, CPM, inventory availability, demographics, screen visibility, hourly traffic, live traffic, or campaign ROI.
- Surface missing operational inputs instead of filling them with assumptions.
- Treat `field_validation_candidates` as a screening result; never present the historical-anchor universe as live ZAP inventory.
- Reject coordinates outside the supported Amsterdam bounds reported by the runtime.

## Modify or validate the implementation

When changing model-facing, shortlist, CLI, or Dashboard logic:

1. Update the deterministic core before changing explanations.
2. Preserve the runtime-bundle schema or version it explicitly.
3. Run `make verify`.
4. Require browser/Python parity tests and shortlist contract tests to pass.
