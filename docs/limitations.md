# Decision Gaps and Field Validation

The model supports early site screening. A media-buying decision still needs information that is not available for arbitrary coordinates in the current public data.

## Missing Decision Inputs

| Missing input | Why it matters | Practical next step |
| --- | --- | --- |
| Screen direction and orientation | Total street traffic is not the same as traffic facing the display | Observe approach direction on both sides of the street |
| Sight line and obstruction | Buildings, trees, street furniture, and other screens affect viewability | Record viewing distance and blocked angles during a site visit |
| Dwell time and crossing behavior | A slower audience may be more valuable than a larger passing flow | Observe crossings, queues, stops, and waiting areas |
| Inventory and price | Exposure alone cannot determine value for money | Join the shortlist with availability, format, and CPM data |
| Audience profile | Weekly passers do not describe age, intent, or campaign fit | Add privacy-safe audience or survey data where permitted |
| Current local conditions | Historical counts may not reflect recent construction or route changes | Run short manual counts at relevant times before purchase |

## How to Use the Current Output

1. Use predicted weekly passers to screen and order candidate locations.
2. Use prediction intervals and caution signals to decide where validation matters most.
3. Compare the model estimate with short on-site counts and local route observations.
4. Add commercial and screen-level data before making a final media decision.

The formal model risks and transfer limits are documented in the [model card](model_card.md). This page focuses only on the information still needed for an operational decision.
