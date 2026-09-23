# Product Brief

## Product

Amsterdam OOH Location Intelligence for early-stage advertising site selection.

## Target User

Commercial analysts, media planners, and expansion teams at an outdoor advertising company.

## User Problem

Teams need to prioritize candidate advertising locations before fieldwork or campaign testing. The available data is fragmented, and decisions often depend on subjective local knowledge.

## Core Jobs To Be Done

- Estimate typical weekly exposure for a selected candidate location.
- Save and compare multiple candidate locations.
- Understand why a location ranks highly.
- Identify sites with high uncertainty.
- Compare the selected location with its most similar observed anchors.
- Generate a campaign-specific shortlist using explicit context and evidence rules.

## Core User Stories

- As a media planner, I want to see the top locations by predicted weekly passers so that I can shortlist sites quickly.
- As a media planner, I want to input a candidate screen coordinate so that I can estimate its likely weekly exposure before field validation.
- As a commercial analyst, I want to inspect retail, horeca, transit, and tourism profiles so that I can explain recommendations to stakeholders.
- As a manager, I want uncertainty flags so that I know which sites need field validation.
- As a user, I want to ask scoped natural-language questions about the selected location so that I can understand the recommendation and its limits faster.

## Success Metrics

- The tool generates a reproducible shortlist from a saved candidate set.
- Each recommendation includes a clear evidence trail.
- Model outputs are reproducible from cached data.
- The dashboard and assistant do not claim more precision or business coverage than the data supports.

## Agent Boundary

The V2 bounded agent can invoke explicit tools to save, compare, filter, rank, and explain candidates. Campaign objectives are applied as transparent eligibility conditions before eligible sites are ranked by predicted weekly exposure. Model values and evidence remain deterministic.

A later version may replace the deterministic intent parser with a language-model planner, but only through the same registered tool schemas. Inventory price, availability, audience demographics, screen visibility, and campaign ROI remain outside scope until suitable data sources are connected.
