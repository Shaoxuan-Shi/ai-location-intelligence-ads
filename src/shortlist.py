from __future__ import annotations

import re
from typing import Any


OBJECTIVE_ALIASES = {
    "general": "exposure",
    "maximum": "exposure",
    "shopping": "retail",
    "commuter": "transit",
}


def normalize_brief(brief: dict[str, Any] | None = None) -> dict[str, Any]:
    source = brief or {}
    objective = OBJECTIVE_ALIASES.get(str(source.get("objective", "exposure")), str(source.get("objective", "exposure")))
    if objective not in {"exposure", "tourism", "retail", "transit"}:
        objective = "exposure"
    evidence_policy = str(source.get("evidence_policy", source.get("evidencePolicy", "balanced")))
    if evidence_policy not in {"strict", "balanced", "exploratory"}:
        evidence_policy = "balanced"
    shortlist_size = max(1, min(8, int(source.get("shortlist_size", source.get("shortlistSize", 3)) or 3)))
    return {"objective": objective, "evidence_policy": evidence_policy, "shortlist_size": shortlist_size}


def parse_campaign_brief(query: str) -> dict[str, Any]:
    q = query.lower()
    objective = "exposure"
    if re.search(r"touris|visitor|museum|hotel", q):
        objective = "tourism"
    elif re.search(r"retail|shopping|shop|store|fashion", q):
        objective = "retail"
    elif re.search(r"transit|commut|station|metro|tram", q):
        objective = "transit"
    evidence_policy = "balanced"
    if "strict" in q:
        evidence_policy = "strict"
    elif re.search(r"explor|relax", q):
        evidence_policy = "exploratory"
    numeric = re.search(r"(?:top|shortlist(?: of)?|select|choose)\s+(?:a\s+)?([1-8])(?:\s|-|$)", q)
    return normalize_brief(
        {
            "objective": objective,
            "evidence_policy": evidence_policy,
            "shortlist_size": int(numeric.group(1)) if numeric else 5,
        }
    )


def _number(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in {None, ""}:
            return float(row[key])
    return default


def objective_reasons(row: dict[str, Any], brief: dict[str, Any]) -> list[str]:
    objective = brief["objective"]
    fields = {
        "tourism": ("tourism_percentile", "tourismPct"),
        "retail": ("retail_percentile", "retailPct"),
        "transit": ("transit_percentile", "transitPct"),
    }
    if objective == "exposure":
        return []
    value = _number(row, *fields[objective])
    return [] if value >= 60 else [f"{objective} context is percentile {round(value)}, below 60"]


def evidence_reasons(row: dict[str, Any], brief: dict[str, Any]) -> list[str]:
    policy = brief["evidence_policy"]
    if policy == "exploratory":
        return []
    outside = _number(row, "outside_feature_count", "outsideFeatureCount")
    error = _number(row, "similar_anchor_validation_error", "comparableError")
    variability = _number(row, "comparable_anchor_variability", "variability")
    similarity = _number(row, "joint_feature_similarity_percentile", "jointSimilarityPct")
    support = _number(row, "feature_support_count", "supportCount")
    reasons = []
    if policy == "strict":
        if outside > 0:
            reasons.append(f"{round(outside)} inputs outside training range")
        if error > 0.75:
            reasons.append(f"{round(error * 100)}% similar-anchor held-out error exceeds 75%")
        if variability > 0.45:
            reasons.append(f"{round(variability * 100)}% historical spread exceeds 45%")
        if similarity < 25:
            reasons.append(f"{round(similarity)}% joint similarity is below 25%")
        if support == 0:
            reasons.append("no anchor falls within the typical feature-match threshold")
    else:
        if outside > 1:
            reasons.append(f"{round(outside)} inputs outside training range")
        if error > 1.5:
            reasons.append(f"{round(error * 100)}% similar-anchor held-out error exceeds 150%")
        if similarity < 10:
            reasons.append(f"{round(similarity)}% joint similarity is below 10%")
    return reasons


def build_shortlist(candidates: list[dict[str, Any]], brief: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_brief(brief)
    eligible = []
    excluded = []
    for candidate in candidates:
        reasons = objective_reasons(candidate, normalized) + evidence_reasons(candidate, normalized)
        if reasons:
            excluded.append({"candidate": candidate, "reasons": reasons})
        else:
            eligible.append(candidate)
    ranked = sorted(
        eligible,
        key=lambda row: _number(row, "predicted_weekly_passers", "predicted"),
        reverse=True,
    )
    return {
        "brief": normalized,
        "eligible": eligible,
        "excluded": excluded,
        "ranked": ranked,
        "selected": ranked[: normalized["shortlist_size"]],
    }
