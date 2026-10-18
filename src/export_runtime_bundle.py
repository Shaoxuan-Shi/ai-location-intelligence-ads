from __future__ import annotations

import json
import re
from pathlib import Path

from utils import APP_DIR, OUTPUT_TABLES, write_compact_json


def extract_context(html: str) -> dict:
    match = re.search(
        r'<script id="dashboardContext" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("dashboardContext JSON block was not found")
    return json.loads(match.group(1))


def add_runtime_metadata(context: dict) -> dict:
    audit = context.get("audit", {})
    model_name = next(
        (row.get("value") for row in context.get("validationSummary", []) if row.get("metric") == "selected_model"),
        "footfall-model",
    )
    data_as_of = str(audit.get("last_week", "unknown"))[:10]
    context["runtime"] = {
        "schema_version": "1.0.0",
        "model_version": f"{model_name}-{data_as_of}-v1",
        "data_as_of": data_as_of,
        "geography": "Amsterdam",
        "target": "typical weekly pedestrian exposure",
    }
    return context


def main() -> None:
    source = APP_DIR / "dashboard_agent.html"
    destination = OUTPUT_TABLES.parent / "runtime" / "runtime_bundle.json"
    context = add_runtime_metadata(extract_context(source.read_text(encoding="utf-8")))
    write_compact_json(destination, context)
    print(destination)


if __name__ == "__main__":
    main()
