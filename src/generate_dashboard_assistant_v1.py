from __future__ import annotations

import json
import math
from typing import Any

from build_features import load_horeca_points, load_osm_points
from utils import APP_DIR, DATA_RAW, OUTPUT_TABLES, html_safe_json, quantile, read_csv, read_json, safe_float


def compact_anchor(row: dict[str, str]) -> dict[str, object]:
    return {
        "name": row["location_name"],
        "lat": round(safe_float(row["lat"]), 6),
        "lon": round(safe_float(row["lon"]), 6),
        "actual": round(safe_float(row["weekly_passers"])),
        "mean": round(safe_float(row.get("mean_weekly_passers"))),
        "p25": round(safe_float(row.get("p25_weekly_passers"))),
        "p75": round(safe_float(row.get("p75_weekly_passers"))),
        "variability": safe_float(row.get("variability_ratio")),
        "predicted": round(safe_float(row["predicted_weekly_passers"])),
        "looPredicted": round(safe_float(row.get("loocv_predicted_weekly_passers"))),
        "low": round(safe_float(row["prediction_low"])),
        "high": round(safe_float(row["prediction_high"])),
        "rank": int(safe_float(row["predicted_rank"])),
        "shoppingArea": row["shopping_area_name"],
        "retailPct": safe_float(row["shop_count_80m_percentile"]),
        "horecaPct": safe_float(row["horeca_count_120m_percentile"]),
        "transitPct": safe_float(row["transit_stop_count_250m_percentile"]),
        "tourismPct": safe_float(row["tourism_poi_count_300m_percentile"]),
        "nearbyAnchorPct": safe_float(row.get("nearby_anchor_median_passers_percentile")),
        "retail": round(safe_float(row["shop_count_80m"])),
        "horeca": round(safe_float(row["horeca_count_120m"])),
        "transit": round(safe_float(row["transit_stop_count_250m"])),
        "nearestTransit": safe_float(row["distance_to_nearest_transit_m"]),
        "tourism": round(safe_float(row["tourism_poi_count_300m"])),
        "nearbyAnchorExposure": round(safe_float(row.get("nearby_anchor_median_passers"))),
        "nearestAnchorDistance": safe_float(row.get("nearest_anchor_distance_m")),
        "nearestAnchorName": row.get("nearest_anchor_name", ""),
        "looPctError": safe_float(row.get("loocv_absolute_percentage_error")) / 100,
        "intervalWidthRatio": safe_float(row.get("interval_width_ratio")),
        "featureMaxAbsZ": safe_float(row.get("feature_max_abs_z")),
        "rawFeatures": {
            "shop_count_80m": safe_float(row["shop_count_80m"]),
            "horeca_count_120m": safe_float(row["horeca_count_120m"]),
            "transit_stop_count_250m": safe_float(row["transit_stop_count_250m"]),
            "distance_to_nearest_transit_m": safe_float(row["distance_to_nearest_transit_m"]),
            "tourism_poi_count_300m": safe_float(row["tourism_poi_count_300m"]),
            "in_official_shopping_area": safe_float(row["in_official_shopping_area"]),
            "high_street_shop_count_120m": safe_float(row.get("high_street_shop_count_120m")),
            "in_key_shopping_area": safe_float(row.get("in_key_shopping_area")),
            "lodging_count_500m": safe_float(row.get("lodging_count_500m")),
            "horeca_count_200m": safe_float(row.get("horeca_count_200m")),
            "nearby_anchor_median_passers": safe_float(row.get("nearby_anchor_median_passers")),
        },
    }


def compact_pois() -> list[list[object]]:
    rows: list[list[object]] = []
    for point in load_osm_points():
        subtype = point.get("shop_type") or point.get("tourism_type") or point.get("transit_type") or ""
        rows.append([round(float(point["lat"]), 6), round(float(point["lon"]), 6), point["category"], subtype])
    for point in load_horeca_points():
        rows.append([round(float(point["lat"]), 6), round(float(point["lon"]), 6), "horeca", point.get("category", "")])
    return rows


def compact_shopping_areas() -> list[dict[str, Any]]:
    data = read_json(DATA_RAW / "winkelgebieden.geojson")
    areas = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        areas.append(
            {
                "name": props.get("gebiedsnaam") or "",
                "category": props.get("categorie") or "",
                "geometry": feature.get("geometry"),
            }
        )
    return areas


def training_values(anchors: list[dict[str, object]]) -> dict[str, list[float]]:
    mapping = {
        "shop_count_80m": "retail",
        "shop_count_120m": "shop_count_120m",
        "horeca_count_120m": "horeca",
        "transit_stop_count_250m": "transit",
        "distance_to_nearest_transit_m": "nearestTransit",
        "tourism_poi_count_300m": "tourism",
        "nearby_anchor_median_passers": "nearbyAnchorExposure",
    }
    feature_rows = read_csv(OUTPUT_TABLES.parent.parent / "data" / "processed" / "site_features_weekly.csv")
    values: dict[str, list[float]] = {}
    for raw_name in mapping:
        if raw_name == "nearby_anchor_median_passers":
            values[raw_name] = [safe_float(anchor["nearbyAnchorExposure"]) for anchor in anchors]
        else:
            values[raw_name] = [safe_float(row[raw_name]) for row in feature_rows]
    return values


def model_feature_ranges(feature_rows: list[dict[str, str]], artifact: dict[str, Any]) -> dict[str, dict[str, float]]:
    ranges = {}
    for spec in artifact["features"]:
        values = [safe_float(row.get(spec["name"])) for row in feature_rows]
        ranges[spec["name"]] = {"min": min(values), "max": max(values)}
    return ranges


def numeric_range(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {"min": min(ordered), "median": median, "max": max(ordered)}


def feature_support_summary(anchors: list[dict[str, object]], artifact: dict[str, Any]) -> dict[str, Any]:
    vectors = []
    for anchor in anchors:
        values = []
        raw_features = anchor["rawFeatures"]
        for spec in artifact["features"]:
            value = safe_float(raw_features.get(spec["name"]))
            transformed = value if spec.get("transform") == "identity" else math.log1p(max(value, 0.0))
            values.append((transformed - safe_float(spec["mean"])) / (safe_float(spec["std"], 1.0) or 1.0))
        vectors.append(values)

    nearest_distances = []
    for index, vector in enumerate(vectors):
        distances = [
            (sum((left - right) ** 2 for left, right in zip(vector, other)) / len(vector)) ** 0.5
            for other_index, other in enumerate(vectors)
            if other_index != index
        ]
        nearest_distances.append(min(distances))
    return {
        "nearestDistances": nearest_distances,
        "typicalThreshold": quantile(nearest_distances, 0.75),
    }


def main() -> None:
    anchors = [compact_anchor(row) for row in read_csv(OUTPUT_TABLES / "site_predictions_weekly.csv")]
    anchors.sort(key=lambda row: row["rank"])
    artifact = read_json(OUTPUT_TABLES / "model_artifact.json")
    metrics = read_csv(OUTPUT_TABLES / "model_metrics.csv")
    validation_summary = read_csv(OUTPUT_TABLES / "model_validation_summary.csv")
    residuals = read_csv(OUTPUT_TABLES / "largest_residuals.csv")
    feature_rows = read_csv(OUTPUT_TABLES.parent.parent / "data" / "processed" / "site_features_weekly.csv")
    audit = read_csv(OUTPUT_TABLES.parent.parent / "data" / "processed" / "crowdmonitor_weekly_audit.csv")
    context = {
        "anchors": anchors,
        "pois": compact_pois(),
        "shoppingAreas": compact_shopping_areas(),
        "model": {
            "coefficients": artifact["coefficients"],
            "features": artifact["features"],
            "rmseLogFit": artifact["rmse_log_fit"],
            "intervalLogMargin": artifact.get("interval_log_margin", artifact["rmse_log_fit"]),
            "intervalMethod": artifact.get("interval_method", "log residual margin"),
        },
        "modelFeatureRanges": model_feature_ranges(feature_rows, artifact),
        "evidenceRanges": {
            "looError": numeric_range([safe_float(anchor["looPctError"]) for anchor in anchors]),
            "variability": numeric_range([safe_float(anchor["variability"]) for anchor in anchors]),
        },
        "featureSupport": feature_support_summary(anchors, artifact),
        "trainingValues": training_values(anchors),
        "metrics": metrics[:6],
        "validationSummary": validation_summary,
        "residuals": residuals,
        "audit": {row["metric"]: row["value"] for row in audit},
    }
    html_template = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Amsterdam AI Location Intelligence</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha384-sHL9NAb7lN7rfvG5lfHpm643Xkcjzp4jFvuavGOndn6pjVqS6ny56CAt3nsEVT4H" crossorigin="anonymous">
  <link rel="stylesheet" href="https://unpkg.com/maplibre-gl@5.6.1/dist/maplibre-gl.css" integrity="sha384-Nq6PQ+9vJPvw7U/VfDELyrWoGQMsy0gi6QShhaSrGzkpF5KkM40csg2leky+YMTd" crossorigin="anonymous">
  <style>
    :root {
      --bg: #f5f6f1;
      --panel: #ffffff;
      --ink: #1e2526;
      --muted: #617071;
      --line: #d8ddd4;
      --accent: #0b7668;
      --warning: #a35f13;
      --risk: #a33c2f;
      --soft: #e7f1ed;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      padding: 16px 20px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1, h2, h3 { margin: 0; font-weight: 500; }
    h1 { font-size: 23px; }
    h2 { font-size: 18px; margin-bottom: 10px; }
    h3 { font-size: 15px; margin-bottom: 8px; }
    .subtitle { color: var(--muted); margin: 4px 0 0; font-size: 14px; }
    main {
      display: grid;
      grid-template-columns: minmax(380px, 1fr) 470px;
      min-height: calc(100vh - 74px);
    }
    #map { min-height: 720px; width: 100%; }
    .map-load-error {
      position: absolute;
      z-index: 1000;
      top: 12px;
      left: 50%;
      transform: translateX(-50%);
      max-width: min(520px, calc(100% - 96px));
      padding: 9px 12px;
      border: 1px solid var(--warning);
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.96);
      color: var(--ink);
      font-size: 12px;
      box-shadow: 0 4px 16px rgba(30, 37, 38, 0.12);
    }
    aside {
      border-left: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
      overflow: auto;
      max-height: calc(100vh - 74px);
    }
    section {
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
      margin-bottom: 14px;
    }
    .inputs {
      display: grid;
      grid-template-columns: 1.1fr 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    input, textarea, select, button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      font: inherit;
      color: var(--ink);
      background: var(--panel);
    }
    textarea { min-height: 76px; resize: vertical; }
    button {
      cursor: pointer;
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      font-weight: 500;
    }
    .ghost {
      background: var(--panel);
      color: var(--accent);
    }
    .button-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .result-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--bg);
      margin-top: 10px;
    }
    .metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 10px 0;
    }
    .metric {
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .metric strong {
      display: block;
      margin-top: 2px;
      font-size: 17px;
      font-weight: 500;
    }
    .bars {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 78px 1fr 42px;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }
    .bar {
      height: 8px;
      background: #dfe8e3;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar span { display: block; height: 100%; background: var(--accent); }
    .pill {
      display: inline-block;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 12px;
      color: var(--accent);
      background: var(--soft);
    }
    .answer {
      white-space: pre-wrap;
      background: var(--bg);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      color: var(--ink);
      font-size: 13px;
      margin-top: 8px;
      line-height: 1.45;
    }
    .answer:empty { display: none; }
    .assistant-scope {
      color: var(--muted);
      font-size: 12px;
      margin: -3px 0 9px;
    }
    .suggestions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }
    .suggestion {
      width: auto;
      padding: 6px 8px;
      border-color: var(--line);
      background: var(--panel);
      color: var(--accent);
      font-size: 12px;
      font-weight: 500;
      text-align: left;
    }
    .suggestion:hover, .suggestion:focus-visible {
      border-color: var(--accent);
      background: var(--soft);
    }
    .evidence-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 10px;
    }
    .evidence {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: var(--panel);
    }
    .evidence span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .evidence strong {
      display: block;
      margin-top: 2px;
      font-size: 17px;
      font-weight: 500;
    }
    .table-wrap { overflow-x: auto; }
    .anchor-table { max-height: 360px; overflow: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 5px;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 500; }
    tr { cursor: pointer; }
    tr.selected { background: var(--soft); }
    .note { color: var(--muted); font-size: 12px; margin: 6px 0 0; }
    @media (max-width: 940px) {
      main { grid-template-columns: 1fr; }
      #map { min-height: 460px; }
      aside { border-left: none; border-top: 1px solid var(--line); max-height: none; }
    }
    @media (max-width: 520px) {
      .inputs, .button-row, .metrics { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Amsterdam AI Location Intelligence</h1>
    <p class="subtitle">Evaluate outdoor advertising candidate locations by typical weekly exposure</p>
  </header>
  <main>
    <div id="map" aria-label="Map of training anchors and custom candidate"></div>
    <aside>
      <section>
        <h2>Location Evaluator</h2>
        <div class="inputs">
          <label>Name<input id="candidateName" value="Custom candidate"></label>
          <label>Latitude<input id="latInput" value="52.370216"></label>
          <label>Longitude<input id="lonInput" value="4.895168"></label>
        </div>
        <div class="button-row">
          <button id="evaluateBtn">Evaluate Location</button>
          <button class="ghost" id="sampleBtn">Use Kalverstraat</button>
        </div>
        <p class="note">Use Amsterdam coordinates roughly within latitude 52.30-52.43 and longitude 4.75-5.05. Clicking the map fills the coordinates.</p>
        <div id="candidateResult" class="result-card"></div>
      </section>
      <section>
        <h2>Ask About This Location</h2>
        <p class="assistant-scope">Ask about reliability, traffic drivers, comparable anchors, and shortlist suitability.</p>
        <div id="suggestions" class="suggestions" aria-label="Suggested questions"></div>
        <textarea id="askInput" placeholder="Ask a question about the selected location"></textarea>
        <button id="askBtn">Ask</button>
        <div id="answer" class="answer" aria-live="polite"></div>
      </section>
      <section>
        <h2>Training Anchors</h2>
        <p class="note" id="anchorValidationNote"></p>
        <div class="table-wrap anchor-table"><table>
          <thead><tr><th>Anchor</th><th>Observed</th><th>Held-out prediction</th><th title="Absolute percentage error when this anchor is excluded from training">Held-out error</th></tr></thead>
          <tbody id="ranking"></tbody>
        </table></div>
      </section>
      <section>
        <h2>Model Validation</h2>
        <p class="note" id="audit"></p>
        <div id="validationSummary" class="evidence-grid"></div>
        <table>
          <thead><tr><th>Model</th><th>MAE</th><th>MAPE</th><th>RMSE log</th></tr></thead>
          <tbody id="metrics"></tbody>
        </table>
        <p class="note" id="weakCases"></p>
      </section>
    </aside>
  </main>
  <script id="dashboardContext" type="application/json">__CONTEXT__</script>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha384-cxOPjt7s7Iz04uaHJceBmS+qpjv2JkIHNVcuOrM+YHwZOmJGBXI00mdUXEq65HTH" crossorigin="anonymous"></script>
  <script src="https://unpkg.com/maplibre-gl@5.6.1/dist/maplibre-gl.js" integrity="sha384-/L1njH4bbgNt9Uk3HwJ272N9fxJzRBQCxhtwGkZiqgl+Nxpq2ETUNZhNMNV1RgyW" crossorigin="anonymous"></script>
  <script src="https://unpkg.com/@maplibre/maplibre-gl-leaflet@0.1.4/leaflet-maplibre-gl.js" integrity="sha384-tXYNKOHx4T02jMP7YYCtBxPIv1B5gaA5mcVPBzqMp6d7VzWzxJgI2aWF/nJLrQdS" crossorigin="anonymous"></script>
  <script>
    const ctx = JSON.parse(document.getElementById('dashboardContext').textContent);
    const anchors = ctx.anchors;
    const pois = ctx.pois;
    const model = ctx.model;
    let current = null;
    let customMarker = null;

    const mapElement = document.getElementById('map');
    function showMapNotice(message) {
      const notice = document.createElement('div');
      notice.className = 'map-load-error';
      notice.textContent = message;
      mapElement.appendChild(notice);
    }
    let map = null;
    if (typeof L === 'object') {
      map = L.map('map').setView([52.3676, 4.9041], 12);
      const supportsWebGL = (() => {
        try {
          const canvas = document.createElement('canvas');
          return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'));
        } catch (error) {
          return false;
        }
      })();
      if (typeof L.maplibreGL === 'function' && supportsWebGL) {
        try {
          L.maplibreGL({
            style: 'https://tiles.openfreemap.org/styles/positron'
          }).addTo(map);
          map.attributionControl.addAttribution(
            '<a href="https://openfreemap.org/" target="_blank" rel="noopener">OpenFreeMap</a> | ' +
            '<a href="https://www.openmaptiles.org/" target="_blank" rel="noopener">OpenMapTiles</a> | ' +
            'Data from <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>'
          );
        } catch (error) {
          showMapNotice('The vector basemap could not start on this device. Location markers and the evaluator remain available.');
        }
      } else if (!supportsWebGL) {
        showMapNotice('The vector basemap needs WebGL, which is unavailable on this device. Location markers and the evaluator remain available.');
      } else {
        showMapNotice('The basemap library could not load. Check the internet connection and refresh; location markers remain available.');
      }
    } else {
      showMapNotice('The map library could not load. The location evaluator and decision assistant remain available.');
    }

    function fmt(value) { return Math.round(value).toLocaleString('en-US'); }
    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, char => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
    }
    function log1p(value) { return Math.log1p(Math.max(Number(value) || 0, 0)); }
    function expm1(value) { return Math.max(Math.expm1(value), 0); }
    function hav(lat1, lon1, lat2, lon2) {
      const r = 6371000;
      const dLat = (lat2 - lat1) * Math.PI / 180;
      const dLon = (lon2 - lon1) * Math.PI / 180;
      const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
      return 2 * r * Math.asin(Math.sqrt(a));
    }
    function countNearby(lat, lon, radius, category) {
      let total = 0;
      for (const point of pois) {
        if (point[2] === category && hav(lat, lon, point[0], point[1]) <= radius) total += 1;
      }
      return total;
    }
    function countMatching(lat, lon, radius, category, subtypes) {
      let total = 0;
      for (const point of pois) {
        if (point[2] === category && subtypes.includes(point[3]) && hav(lat, lon, point[0], point[1]) <= radius) total += 1;
      }
      return total;
    }
    function nearestTransit(lat, lon) {
      let best = 9999;
      for (const point of pois) {
        if (point[2] === 'transit') best = Math.min(best, hav(lat, lon, point[0], point[1]));
      }
      return best;
    }
    function nearbyAnchorExposure(lat, lon) {
      let weighted = 0;
      let total = 0;
      let nearest = {name: '', distance: 9999};
      for (const anchor of anchors) {
        const distance = hav(lat, lon, anchor.lat, anchor.lon);
        if (distance < nearest.distance) nearest = {name: anchor.name, distance};
        const weight = 1 / Math.max(distance, 80);
        weighted += anchor.actual * weight;
        total += weight;
      }
      return {
        exposure: total ? weighted / total : 0,
        nearestName: nearest.name,
        nearestDistance: nearest.distance
      };
    }
    function inRing(lon, lat, ring) {
      let inside = false;
      for (let i = 0; i < ring.length; i++) {
        const [x1, y1] = ring[i];
        const [x2, y2] = ring[(i + 1) % ring.length];
        if (((y1 > lat) !== (y2 > lat)) && lon < (x2 - x1) * (lat - y1) / ((y2 - y1) || 1e-15) + x1) inside = !inside;
      }
      return inside;
    }
    function inPolygon(lon, lat, polygon) {
      if (!polygon.length || !inRing(lon, lat, polygon[0])) return false;
      return !polygon.slice(1).some(ring => inRing(lon, lat, ring));
    }
    function shoppingArea(lat, lon) {
      for (const area of ctx.shoppingAreas) {
        const geom = area.geometry;
        if (!geom) continue;
        if (geom.type === 'Polygon' && inPolygon(lon, lat, geom.coordinates)) return area;
        if (geom.type === 'MultiPolygon' && geom.coordinates.some(poly => inPolygon(lon, lat, poly))) return area;
      }
      return {name: '', category: ''};
    }
    function percentile(value, values, higher = true) {
      let lower = 0, equal = 0;
      for (const item of values) {
        if (item < value) lower += 1;
        if (item === value) equal += 1;
      }
      let pct = 100 * (lower + 0.5 * equal) / values.length;
      if (!higher) pct = 100 - pct;
      return Math.round(pct * 10) / 10;
    }
    function rawFeatureRow(lat, lon, name) {
      const area = shoppingArea(lat, lon);
      const nearby = nearbyAnchorExposure(lat, lon);
      const highStreetTypes = ['clothes', 'shoes', 'jewelry', 'department_store', 'gift', 'cosmetics', 'fashion_accessories'];
      const lodgingTypes = ['hotel', 'hostel', 'guest_house', 'apartment', 'motel'];
      const row = {
        name,
        lat,
        lon,
        retail: countNearby(lat, lon, 80, 'shop'),
        shop120: countNearby(lat, lon, 120, 'shop'),
        horeca: countNearby(lat, lon, 120, 'horeca'),
        horeca200: countNearby(lat, lon, 200, 'horeca'),
        transit: countNearby(lat, lon, 250, 'transit'),
        nearestTransit: Math.round(nearestTransit(lat, lon) * 10) / 10,
        tourism: countNearby(lat, lon, 300, 'tourism'),
        highStreetRetail: countMatching(lat, lon, 120, 'shop', highStreetTypes),
        lodging: countMatching(lat, lon, 500, 'tourism', lodgingTypes),
        nearbyAnchorExposure: Math.round(nearby.exposure),
        nearestAnchorName: nearby.nearestName,
        nearestAnchorDistance: Math.round(nearby.nearestDistance * 10) / 10,
        inShoppingArea: area.name ? 1 : 0,
        inKeyShoppingArea: area.category === 'K' ? 1 : 0,
        shoppingArea: area.name,
        shoppingAreaCategory: area.category
      };
      row.rawFeatures = {
        shop_count_80m: row.retail,
        horeca_count_120m: row.horeca,
        transit_stop_count_250m: row.transit,
        distance_to_nearest_transit_m: row.nearestTransit,
        tourism_poi_count_300m: row.tourism,
        in_official_shopping_area: row.inShoppingArea,
        high_street_shop_count_120m: row.highStreetRetail,
        in_key_shopping_area: row.inKeyShoppingArea,
        lodging_count_500m: row.lodging,
        horeca_count_200m: row.horeca200
      };
      row.retailPct = percentile(row.retail, ctx.trainingValues.shop_count_80m, true);
      row.horecaPct = percentile(row.horeca, ctx.trainingValues.horeca_count_120m, true);
      row.transitPct = percentile(row.transit, ctx.trainingValues.transit_stop_count_250m, true);
      row.tourismPct = percentile(row.tourism, ctx.trainingValues.tourism_poi_count_300m, true);
      row.nearbyAnchorPct = percentile(row.nearbyAnchorExposure, ctx.trainingValues.nearby_anchor_median_passers, true);
      return row;
    }
    function transform(row) {
      return model.features.map(spec => spec.transform === 'identity'
        ? Number(row.rawFeatures[spec.name] || 0)
        : log1p(row.rawFeatures[spec.name]));
    }
    function standardize(values) {
      return values.map((value, i) => (value - model.features[i].mean) / (model.features[i].std || 1));
    }
    function predictLog(xStd) {
      return model.coefficients[0] + xStd.reduce((sum, value, i) => sum + value * model.coefficients[i + 1], 0);
    }
    function featureDistance(a, b) {
      return Math.sqrt(a.reduce((sum, value, i) => sum + (value - b[i]) ** 2, 0) / a.length);
    }
    function comparableAnchors(xStd) {
      return anchors.map(anchor => {
        const aStd = standardize(transform(anchor));
        return {...anchor, featureDistance: featureDistance(xStd, aStd), geoDistance: hav(current?.lat || anchor.lat, current?.lon || anchor.lon, anchor.lat, anchor.lon)};
      }).sort((a, b) => a.featureDistance - b.featureDistance);
    }
    function evaluate(lat, lon, name) {
      const row = rawFeatureRow(lat, lon, name);
      const xStd = standardize(transform(row));
      const predLog = predictLog(xStd);
      const novelty = Math.max(0, Math.max(...xStd.map(Math.abs)) - 2);
      const intervalScale = model.intervalLogMargin * (1 + 0.25 * novelty);
      const predicted = expm1(predLog);
      const low = expm1(predLog - intervalScale);
      const high = expm1(predLog + intervalScale);
      current = row;
      const allComps = comparableAnchors(xStd);
      const comps = allComps.slice(0, 3);
      const comparableError = comps.reduce((sum, item) => sum + (item.looPctError || 0.75), 0) / comps.length;
      const variability = comps.reduce((sum, item) => sum + (item.variability || 0.25), 0) / comps.length;
      const meanAbsZ = xStd.reduce((sum, value) => sum + Math.abs(value), 0) / xStd.length;
      const maxAbsZ = Math.max(...xStd.map(Math.abs));
      const widthRatio = (high - low) / Math.max(predicted, 1);
      const outsideFeatureCount = model.features.filter(spec => {
        const value = Number(row.rawFeatures[spec.name] || 0);
        const range = ctx.modelFeatureRanges[spec.name];
        return value < range.min || value > range.max;
      }).length;
      const supportCount = allComps.filter(item => item.featureDistance <= ctx.featureSupport.typicalThreshold).length;
      const jointSimilarityPct = percentile(
        allComps[0]?.featureDistance ?? 999,
        ctx.featureSupport.nearestDistances,
        false
      );
      current = {
        ...row,
        predicted,
        low,
        high,
        comparables: comps,
        comparableError,
        variability,
        meanAbsZ,
        maxAbsZ,
        widthRatio,
        outsideFeatureCount,
        supportCount,
        jointSimilarityPct
      };
      return current;
    }
    function bar(labelText, value) {
      return `<div class="bar-row"><span>${labelText}</span><div class="bar"><span style="width:${Math.max(0, Math.min(100, value))}%"></span></div><span>${Math.round(value)}</span></div>`;
    }
    function renderResult(result) {
      const area = result.shoppingArea || 'Outside official shopping area';
      document.getElementById('candidateResult').innerHTML = `
        <h3>${escapeHtml(result.name)}</h3>
        <div class="metrics">
          <div class="metric"><span>Predicted typical weekly passers</span><strong>${fmt(result.predicted)}</strong></div>
          <div class="metric"><span>Approx. 80% prediction interval</span><strong>${fmt(result.low)}-${fmt(result.high)}</strong></div>
          <div class="metric"><span>Nearest comparable anchor</span><strong>${escapeHtml(result.comparables[0]?.name || 'n/a')}</strong></div>
          <div class="metric"><span>Shopping context</span><strong>${escapeHtml(area)}</strong></div>
        </div>
        <h3>Reliability Evidence</h3>
        <div class="evidence-grid">
          <div class="evidence"><span>Comparable-anchor support</span><strong>${result.comparables[0]?.featureDistance.toFixed(2) ?? 'n/a'} nearest feature distance</strong><span>${result.supportCount} anchor${result.supportCount === 1 ? '' : 's'} ${result.supportCount === 1 ? 'is' : 'are'} within the typical training-match threshold of ${ctx.featureSupport.typicalThreshold.toFixed(2)}. Closest is ${escapeHtml(result.comparables[0]?.name || 'n/a')}, ${Math.round(result.comparables[0]?.geoDistance || 0)}m away geographically.</span></div>
          <div class="evidence"><span>Similar-anchor validation error</span><strong>${Math.round(result.comparableError * 100)}% average error</strong><span>The three matches missed their own held-out traffic by this amount. Training range: ${Math.round(ctx.evidenceRanges.looError.min * 100)}%-${Math.round(ctx.evidenceRanges.looError.max * 100)}%; lower strengthens local evidence.</span></div>
          <div class="evidence"><span>Historical stability</span><strong>${Math.round(result.variability * 100)}% weekly spread</strong><span>Comparable anchors' middle 50% of weeks span this share of their median. Training range: ${Math.round(ctx.evidenceRanges.variability.min * 100)}%-${Math.round(ctx.evidenceRanges.variability.max * 100)}%; lower is more stable.</span></div>
          <div class="evidence"><span>Training range check</span><strong>${result.outsideFeatureCount} of ${model.features.length} inputs outside range</strong><span>Joint feature similarity is stronger than ${Math.round(result.jointSimilarityPct)}% of training anchors' nearest matches. The most extreme single input is ${result.maxAbsZ.toFixed(2)} standard deviations from the training mean.</span></div>
        </div>
        <div class="bars">${bar('Retail', result.retailPct)}${bar('Horeca', result.horecaPct)}${bar('Transit', result.transitPct)}${bar('Tourism', result.tourismPct)}</div>
        <p class="note">Comparable anchors: ${result.comparables.map(item => `${escapeHtml(item.name)} (${fmt(item.actual)}, ${Math.round(item.geoDistance)}m)`).join(', ')}</p>
      `;
    }
    function placeCustomMarker(result) {
      if (!map || typeof L !== 'object') return;
      if (customMarker) customMarker.remove();
      customMarker = L.marker([result.lat, result.lon]).addTo(map);
      customMarker.bindPopup(`<strong>${escapeHtml(result.name)}</strong><br>Predicted weekly passers: ${fmt(result.predicted)}<br>Interval: ${fmt(result.low)}-${fmt(result.high)}`).openPopup();
    }
    function isSupportedCoordinate(lat, lon) {
      return Number.isFinite(lat) && Number.isFinite(lon) && lat >= 52.30 && lat <= 52.43 && lon >= 4.75 && lon <= 5.05;
    }
    function runEvaluation() {
      const lat = Number(document.getElementById('latInput').value);
      const lon = Number(document.getElementById('lonInput').value);
      const name = document.getElementById('candidateName').value || 'Custom candidate';
      if (!isSupportedCoordinate(lat, lon)) {
        document.getElementById('answer').textContent = 'Enter a coordinate inside the supported Amsterdam area: latitude 52.30-52.43 and longitude 4.75-5.05.';
        return;
      }
      const result = evaluate(lat, lon, name);
      renderResult(result);
      placeCustomMarker(result);
      renderSuggestions(result);
      document.getElementById('answer').textContent = '';
    }
    const featureMeta = {
      shop_count_80m: {label: 'Nearby retail', value: row => `${row.retail} shops within 80m`},
      horeca_count_120m: {label: 'Immediate horeca', value: row => `${row.horeca} venues within 120m`},
      transit_stop_count_250m: {label: 'Nearby transit', value: row => `${row.transit} stops within 250m`},
      distance_to_nearest_transit_m: {label: 'Transit distance', value: row => `${Math.round(row.nearestTransit)}m to the nearest stop`},
      tourism_poi_count_300m: {label: 'Tourism context', value: row => `${row.tourism} tourism or historic POIs within 300m`},
      in_official_shopping_area: {label: 'Shopping-area status', value: row => row.inShoppingArea ? 'inside an official shopping area' : 'outside an official shopping area'},
      high_street_shop_count_120m: {label: 'High-street retail', value: row => `${row.highStreetRetail} high-street shops within 120m`},
      in_key_shopping_area: {label: 'Key shopping district', value: row => row.inKeyShoppingArea ? 'inside a key shopping district' : 'outside a key shopping district'},
      lodging_count_500m: {label: 'Lodging context', value: row => `${row.lodging} lodging POIs within 500m`},
      horeca_count_200m: {label: 'Wider horeca', value: row => `${row.horeca200} venues within 200m`}
    };
    function contributionDirection(value) {
      if (value > 0.04) return 'pushes the estimate upward';
      if (value < -0.04) return 'pulls the estimate downward';
      return 'has little effect on the estimate';
    }
    function modelContributions(row) {
      const xStd = standardize(transform(row));
      return model.features.map((spec, index) => ({
        name: spec.name,
        label: featureMeta[spec.name]?.label || spec.name.replaceAll('_', ' '),
        value: featureMeta[spec.name]?.value(row) || String(row.rawFeatures[spec.name] || 0),
        contribution: xStd[index] * model.coefficients[index + 1]
      })).sort((left, right) => Math.abs(right.contribution) - Math.abs(left.contribution));
    }
    function outOfRangeFeatures(row) {
      return model.features.filter(spec => {
        const value = Number(row.rawFeatures[spec.name] || 0);
        const range = ctx.modelFeatureRanges[spec.name];
        return value < range.min || value > range.max;
      }).map(spec => {
        const value = Number(row.rawFeatures[spec.name] || 0);
        const range = ctx.modelFeatureRanges[spec.name];
        return {
          label: featureMeta[spec.name]?.label || spec.name.replaceAll('_', ' '),
          value,
          min: range.min,
          max: range.max
        };
      });
    }
    function reliabilityFlags(row) {
      const flags = [];
      if (row.comparableError > 0.75) flags.push('similar-anchor validation error');
      if (row.variability > 0.45) flags.push('historical variability');
      if (row.outsideFeatureCount > 0) flags.push('inputs outside the training range');
      if (row.supportCount === 0 || row.jointSimilarityPct < 25) flags.push('weak joint feature similarity');
      return flags;
    }
    function renderSuggestions(row) {
      const flags = reliabilityFlags(row);
      const questions = [
        'Why is this estimate high or low?',
        row.outsideFeatureCount > 0
          ? 'Which features are outside the training range?'
          : flags.length
            ? 'Why is the local evidence weak?'
            : 'How reliable is this prediction?',
        'Which anchors are most comparable?',
        'Should this location be shortlisted?'
      ];
      const container = document.getElementById('suggestions');
      container.innerHTML = '';
      questions.forEach(question => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'suggestion';
        button.textContent = question;
        button.addEventListener('click', () => {
          document.getElementById('askInput').value = question;
          answerQuestion();
        });
        container.appendChild(button);
      });
    }
    function addSection(sections, title, conclusion, evidence = [], action = '') {
      const lines = [`${title}: ${conclusion}`];
      evidence.forEach(item => lines.push(`- ${item}`));
      if (action) lines.push(`Next step: ${action}`);
      sections.push(lines.join('\n'));
    }
    function featureContribution(row, name) {
      return modelContributions(row).find(item => item.name === name);
    }
    function contributionEvidence(item) {
      const signed = `${item.contribution >= 0 ? '+' : ''}${item.contribution.toFixed(2)}`;
      return `${item.label}: ${item.value}; ${contributionDirection(item.contribution)} (${signed} on the model's log scale).`;
    }
    function answerEstimate(sections, row) {
      addSection(
        sections,
        'Exposure estimate',
        `The model estimates ${fmt(row.predicted)} typical weekly passers, with an approximate 80% prediction interval of ${fmt(row.low)}-${fmt(row.high)}.`,
        ['The target is the normal-week median observed across historical Crowdmonitor weeks, not a live count or a daily forecast.']
      );
    }
    function answerDrivers(sections, row) {
      const contributions = modelContributions(row);
      const positive = contributions.filter(item => item.contribution > 0.04).slice(0, 2);
      const negative = contributions.filter(item => item.contribution < -0.04).slice(0, 2);
      const exposurePct = percentile(row.predicted, anchors.map(anchor => anchor.actual), true);
      const position = exposurePct >= 70 ? 'toward the high end' : exposurePct <= 30 ? 'toward the low end' : 'near the middle';
      const evidence = [...positive, ...negative].map(contributionEvidence);
      if (!evidence.length) evidence.push('No individual feature makes a large contribution relative to the average training profile.');
      addSection(
        sections,
        'Main model drivers',
        `The estimate sits ${position} of observed training-anchor traffic, above about ${Math.round(exposurePct)}% of anchors.`,
        evidence,
        'Use these as model associations for site screening, not as proof that any feature causes pedestrian traffic.'
      );
    }
    function answerFeatures(sections, row, requested) {
      const evidence = [];
      if (requested.has('tourism')) {
        const item = featureContribution(row, 'tourism_poi_count_300m');
        evidence.push(`Tourism: ${row.tourism} eligible OSM tourism or historic POIs within 300m, percentile ${Math.round(row.tourismPct)}; ${row.tourism === 0 ? 'zero means the cached source contains no eligible POIs in that radius, not that there are no visitors.' : contributionDirection(item.contribution) + ` (${item.contribution >= 0 ? '+' : ''}${item.contribution.toFixed(2)} log contribution).`}`);
      }
      if (requested.has('retail')) {
        const item = featureContribution(row, 'shop_count_80m');
        evidence.push(`Retail: ${row.retail} shops within 80m, percentile ${Math.round(row.retailPct)}; ${contributionDirection(item.contribution)} (${item.contribution >= 0 ? '+' : ''}${item.contribution.toFixed(2)} log contribution).`);
      }
      if (requested.has('horeca')) {
        const near = featureContribution(row, 'horeca_count_120m');
        const wide = featureContribution(row, 'horeca_count_200m');
        evidence.push(`Horeca: ${row.horeca} venues within 120m and ${row.horeca200} within 200m, percentile ${Math.round(row.horecaPct)}. Their model contributions are ${near.contribution >= 0 ? '+' : ''}${near.contribution.toFixed(2)} and ${wide.contribution >= 0 ? '+' : ''}${wide.contribution.toFixed(2)}.`);
      }
      if (requested.has('transit')) {
        const stops = featureContribution(row, 'transit_stop_count_250m');
        const distance = featureContribution(row, 'distance_to_nearest_transit_m');
        evidence.push(`Transit: ${row.transit} stops within 250m and ${Math.round(row.nearestTransit)}m to the nearest stop, percentile ${Math.round(row.transitPct)}. Their model contributions are ${stops.contribution >= 0 ? '+' : ''}${stops.contribution.toFixed(2)} and ${distance.contribution >= 0 ? '+' : ''}${distance.contribution.toFixed(2)}.`);
      }
      if (requested.has('shopping')) {
        evidence.push(`Shopping area: this point is ${row.inShoppingArea ? `inside ${row.shoppingArea}` : 'outside an official shopping area'} and ${row.inKeyShoppingArea ? 'inside' : 'outside'} a key shopping district.`);
      }
      if (requested.has('lodging')) {
        const item = featureContribution(row, 'lodging_count_500m');
        evidence.push(`Lodging: ${row.lodging} lodging POIs within 500m; ${contributionDirection(item.contribution)} (${item.contribution >= 0 ? '+' : ''}${item.contribution.toFixed(2)} log contribution).`);
      }
      addSection(
        sections,
        'Feature explanation',
        'Each feature is measured from the cached location data and then interpreted through the fitted model.',
        evidence
      );
    }
    function answerReliability(sections, row) {
      const flags = reliabilityFlags(row);
      const conclusion = flags.length === 0
        ? 'The available local evidence is comparatively supportive, although the estimate remains a screening forecast.'
        : flags.length === 1
          ? `One check warrants caution: ${flags[0]}.`
          : `Several checks warrant caution: ${flags.join(', ')}.`;
      addSection(
        sections,
        'Reliability',
        conclusion,
        [
          `Prediction interval: ${fmt(row.low)}-${fmt(row.high)} weekly passers; this describes model uncertainty, not a guaranteed range.`,
          `Comparable anchors averaged ${Math.round(row.comparableError * 100)}% held-out error when each was tested as unseen.`,
          `Comparable anchors' middle 50% of historical weeks span ${Math.round(row.variability * 100)}% of their median; lower is more stable.`,
          `${row.outsideFeatureCount} of ${model.features.length} inputs are outside their observed training range, and joint similarity is stronger than ${Math.round(row.jointSimilarityPct)}% of training anchors' nearest matches.`
        ],
        flags.length ? 'Validate pedestrian flow and screen visibility before committing media budget.' : 'Use the estimate for shortlisting, then validate visibility and pedestrian flow before purchase.'
      );
    }
    function answerTrainingRange(sections, row) {
      const outside = outOfRangeFeatures(row);
      const evidence = outside.length
        ? outside.map(item => `${item.label}: ${item.value} versus training range ${item.min}-${item.max}.`)
        : [`Every individual input falls within its observed training range. Joint similarity is stronger than ${Math.round(row.jointSimilarityPct)}% of training anchors' nearest matches, so in-range inputs do not automatically mean the overall feature combination is familiar.`];
      addSection(
        sections,
        'Training range',
        outside.length ? `${outside.length} model input${outside.length === 1 ? ' is' : 's are'} outside the observed range.` : 'No individual model input is outside its observed range.',
        evidence,
        outside.length ? 'Treat the estimate as extrapolation and inspect the out-of-range inputs before shortlisting.' : 'Use joint similarity and comparable-anchor validation alongside this check.'
      );
    }
    function answerComparables(sections, row) {
      const closest = row.comparables[0];
      const delta = closest?.actual ? 100 * (row.predicted - closest.actual) / closest.actual : 0;
      addSection(
        sections,
        'Comparable anchors',
        `${closest?.name || 'No anchor'} is the closest match in the model's feature space, not necessarily the closest point on the map.`,
        [
          ...row.comparables.map(item => `${item.name}: feature distance ${item.featureDistance.toFixed(2)}, ${Math.round(item.geoDistance)}m away, observed median ${fmt(item.actual)} weekly passers.`),
          `The nearest anchor geographically is ${row.nearestAnchorName}, about ${Math.round(row.nearestAnchorDistance)}m away.`,
          closest ? `This candidate's estimate is ${Math.abs(Math.round(delta))}% ${delta >= 0 ? 'above' : 'below'} the closest feature match's observed median.` : ''
        ].filter(Boolean),
        'Compare local street direction, entrances, visibility, and pedestrian routing before treating an anchor as operationally equivalent.'
      );
    }
    function answerShortlist(sections, row) {
      const exposurePct = percentile(row.predicted, anchors.map(anchor => anchor.actual), true);
      const flags = reliabilityFlags(row);
      let conclusion;
      let action;
      if (exposurePct >= 65) {
        conclusion = flags.length
          ? 'Keep this on an exploratory shortlist: exposure potential is strong, but local evidence needs validation.'
          : 'Shortlist this location for field validation: exposure potential is strong and the available local evidence is supportive.';
        action = 'Check pedestrian direction, screen visibility, dwell time, and commercial availability before media buying.';
      } else if (exposurePct >= 35) {
        conclusion = 'Keep this as a secondary candidate rather than a priority exposure-led site.';
        action = 'Compare it with stronger candidates or retain it only if cost, audience fit, or geographic coverage adds value.';
      } else {
        conclusion = 'Do not prioritize this location for a campaign whose main objective is weekly pedestrian exposure.';
        action = 'Only retain it if a different campaign objective, such as local reach or lower inventory cost, outweighs exposure volume.';
      }
      addSection(
        sections,
        'Shortlist recommendation',
        conclusion,
        [
          `Estimated exposure is above about ${Math.round(exposurePct)}% of observed training anchors.`,
          flags.length ? `Caution signals: ${flags.join(', ')}.` : 'No current local evidence check crosses its caution threshold.',
          'This recommendation assumes weekly pedestrian exposure is the primary campaign objective.'
        ],
        action
      );
    }
    function answerScope(sections) {
      addSection(
        sections,
        'Assistant scope',
        'I can explain the selected location using the evaluator and its model evidence.',
        [
          'Supported: weekly exposure estimates, model drivers, reliability evidence, training-range checks, comparable anchors, and exposure-led shortlist suitability.',
          'Not supported by the current data: hourly or live traffic, audience demographics, CPM, media cost, screen visibility, campaign ROI, and inventory availability.'
        ]
      );
    }
    function answerUnsupported(sections, q) {
      const unsupported = [];
      if (/hour|daily|day of week|live|real.time/.test(q)) unsupported.push('hourly, daily, or live traffic');
      if (/demographic|audience|age|income|gender/.test(q)) unsupported.push('audience demographics');
      if (/cpm|cost|price|budget|roi|return/.test(q)) unsupported.push('media economics or ROI');
      if (/visibility|viewability|screen|billboard|orientation/.test(q)) unsupported.push('screen visibility or orientation');
      if (/availability|inventory|book/.test(q)) unsupported.push('inventory availability');
      if (!unsupported.length) return false;
      addSection(
        sections,
        'Data boundary',
        `The current data cannot determine ${unsupported.join(', ')}.`,
        ['The model estimates typical weekly pedestrian exposure from historical anchors and location context only.'],
        'Use an inventory, audience, or field-audit source for this decision.'
      );
      return true;
    }
    function answerQuestion() {
      if (!current) runEvaluation();
      const input = document.getElementById('askInput');
      const q = input.value.trim().toLowerCase();
      const row = current;
      const sections = [];
      if (!q) {
        answerScope(sections);
        document.getElementById('answer').textContent = sections.join('\n\n');
        return;
      }

      const requestedFeatures = new Set();
      if (/touris|historic|attraction/.test(q)) requestedFeatures.add('tourism');
      if (/retail|shop/.test(q)) requestedFeatures.add('retail');
      if (/horeca|restaurant|bar|cafe/.test(q)) requestedFeatures.add('horeca');
      if (/transit|transport|station|bus|tram|metro/.test(q)) requestedFeatures.add('transit');
      if (/shopping area|shopping district/.test(q)) requestedFeatures.add('shopping');
      if (/lodging|hotel|hostel/.test(q)) requestedFeatures.add('lodging');

      const asksEstimate = /how many|what(?:'s| is) the (?:prediction|estimate)|estimated weekly|predicted weekly|weekly passers|footfall estimate|traffic estimate/.test(q);
      const asksDrivers = /driver|factor|signal|contribution|explain (?:the )?estimate|why (?:is |does )?(?:this|the).*?(?:high|low|estimate|prediction)/.test(q);
      const asksReliability = /reliab|confidence|uncertain|accur|precision|error|trust|local evidence|validate/.test(q);
      const asksRange = /outside.*range|training range|coverage|extrapolat|out.of.range/.test(q);
      const asksComparable = /anchor|comparable|similar location|nearest|distance|how far|compare/.test(q);
      const asksShortlist = /shortlist|recommend|suitable|prioriti|worth|buying media|buy media|good location|campaign/.test(q);
      const asksScope = /what can you|what do you|help me with|answer range|your scope/.test(q);

      if (asksEstimate) answerEstimate(sections, row);
      if (asksDrivers) answerDrivers(sections, row);
      if (requestedFeatures.size) answerFeatures(sections, row, requestedFeatures);
      if (asksReliability) answerReliability(sections, row);
      if (asksRange) answerTrainingRange(sections, row);
      if (asksComparable) answerComparables(sections, row);
      if (asksShortlist) answerShortlist(sections, row);
      if (asksScope) answerScope(sections);
      const hadUnsupported = answerUnsupported(sections, q);
      if (!sections.length && !hadUnsupported) answerScope(sections);
      document.getElementById('answer').textContent = sections.join('\n\n');
    }
    function renderTables() {
      document.getElementById('anchorValidationNote').textContent = `Each row is tested as unseen: the model is trained on the other ${anchors.length - 1} anchors, then predicts this one. Held-out error is |observed - prediction| / observed.`;
      document.getElementById('ranking').innerHTML = anchors.map((anchor, index) => `
        <tr data-anchor-index="${index}">
          <td>${escapeHtml(anchor.name)}</td><td>${fmt(anchor.actual)}</td><td>${fmt(anchor.looPredicted)}</td><td>${Math.round(anchor.looPctError * 100)}%</td>
        </tr>
      `).join('');
      document.querySelectorAll('#ranking tr').forEach(row => {
        row.addEventListener('click', () => {
          const anchor = anchors[Number(row.dataset.anchorIndex)];
          if (!anchor) return;
          document.getElementById('candidateName').value = anchor.name;
          document.getElementById('latInput').value = anchor.lat;
          document.getElementById('lonInput').value = anchor.lon;
          runEvaluation();
        });
      });
      document.getElementById('metrics').innerHTML = ctx.metrics.map(row => `
        <tr><td>${row.model}${Number(row.alpha) ? ` (${row.alpha})` : ''}</td><td>${fmt(Number(row.mae))}</td><td>${Math.round(Number(row.mape_pct))}%</td><td>${Number(row.rmse_log).toFixed(3)}</td></tr>
      `).join('');
      document.getElementById('validationSummary').innerHTML = ctx.validationSummary.map(row => `
        <div class="evidence"><span>${row.metric.replaceAll('_', ' ')}</span><strong>${row.value}</strong><span>${row.detail}</span></div>
      `).join('');
      document.getElementById('weakCases').textContent = `Largest held-out misses: ${ctx.residuals.slice(0, 3).map(row => `${row.location_name} (${row.loocv_absolute_percentage_error}%)`).join(', ')}. These are local model failures, not uncertainty percentages applied to every candidate.`;
      document.getElementById('audit').textContent = `Crowdmonitor weekly history: ${ctx.audit.weekly_dates} weeks, ${ctx.audit.first_week.slice(0, 10)} to ${ctx.audit.last_week.slice(0, 10)}; ${ctx.audit.usable_locations_with_coordinates} usable anchor locations with coordinates.`;
    }
    if (map && typeof L === 'object') {
      anchors.forEach(anchor => {
        const color = anchor.looPctError <= 0.25 ? '#0b7668' : anchor.looPctError <= 0.75 ? '#a35f13' : '#a33c2f';
        const marker = L.circleMarker([anchor.lat, anchor.lon], {
          radius: Math.max(5, Math.min(18, Math.sqrt(anchor.actual) / 42)),
          color,
          weight: 1,
          fillColor: color,
          fillOpacity: 0.58
        }).addTo(map);
        marker.bindPopup(`<strong>${escapeHtml(anchor.name)}</strong><br>Observed median weekly: ${fmt(anchor.actual)}<br>Held-out prediction: ${fmt(anchor.looPredicted)}<br>Held-out error: ${Math.round(anchor.looPctError * 100)}%`);
      });
      if (anchors.length) map.fitBounds(anchors.map(point => [point.lat, point.lon]), {padding: [24, 24]});
      map.on('click', event => {
        document.getElementById('candidateName').value = 'Map-selected candidate';
        document.getElementById('latInput').value = event.latlng.lat.toFixed(6);
        document.getElementById('lonInput').value = event.latlng.lng.toFixed(6);
        runEvaluation();
      });
    }
    document.getElementById('evaluateBtn').addEventListener('click', runEvaluation);
    document.getElementById('sampleBtn').addEventListener('click', () => {
      const sample = anchors.find(item => item.name.includes('Kalverstraat')) || anchors[0];
      document.getElementById('candidateName').value = sample.name;
      document.getElementById('latInput').value = sample.lat;
      document.getElementById('lonInput').value = sample.lon;
      runEvaluation();
    });
    document.getElementById('askBtn').addEventListener('click', answerQuestion);
    document.getElementById('askInput').addEventListener('keydown', event => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        answerQuestion();
      }
    });
    renderTables();
    runEvaluation();
  </script>
</body>
</html>
"""
    html = html_template.replace("__CONTEXT__", html_safe_json(context))
    APP_DIR.mkdir(parents=True, exist_ok=True)
    (APP_DIR / "dashboard.html").write_text(html, encoding="utf-8")
    print("Generated app/dashboard.html")


if __name__ == "__main__":
    main()
