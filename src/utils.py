from __future__ import annotations

import csv
import json
import math
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUT_TABLES = ROOT / "outputs" / "tables"
APP_DIR = ROOT / "app"


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, OUTPUT_TABLES, APP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def urlread(url: str, timeout: int = 60) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ai-location-intelligence-ads/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def fetch_text_cached(url: str, path: Path, *, refresh: bool = False, sleep_s: float = 0.0) -> str:
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    text = urlread(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if sleep_s:
        time.sleep(sleep_s)
    return text


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def html_safe_json(obj: Any) -> str:
    """Serialize JSON for an HTML script-data block without allowing tag breakout."""
    return (
        json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def rd_to_wgs(x: float, y: float) -> tuple[float, float]:
    x0 = 155000.0
    y0 = 463000.0
    dx = (x - x0) * 1e-5
    dy = (y - y0) * 1e-5
    lat = 52.15517440 + (
        (3235.65389 * dy)
        + (-32.58297 * dx**2)
        + (-0.2475 * dy**2)
        + (-0.84978 * dx**2 * dy)
        + (-0.0655 * dy**3)
        + (-0.01709 * dx**2 * dy**2)
        + (-0.00738 * dx)
        + (0.0053 * dx**4)
        + (-0.00039 * dx**2 * dy**3)
        + (0.00033 * dx**4 * dy)
        + (-0.00012 * dx * dy)
    ) / 3600.0
    lon = 5.38720621 + (
        (5260.52916 * dx)
        + (105.94684 * dx * dy)
        + (2.45656 * dx * dy**2)
        + (-0.81885 * dx**3)
        + (0.05594 * dx * dy**3)
        + (-0.05607 * dx**3 * dy)
        + (0.01199 * dy)
        + (-0.00256 * dx**3 * dy**2)
        + (0.00128 * dx * dy**4)
        + (0.00022 * dy**2)
        + (-0.00022 * dx**2)
        + (0.00026 * dx**5)
    ) / 3600.0
    return lat, lon


def parse_rd_point(wkt: str) -> tuple[float, float]:
    match = re.search(r"POINT \(([-0-9.]+) ([-0-9.]+)\)", wkt or "")
    if not match:
        raise ValueError(f"Could not parse RD point from {wkt!r}")
    return float(match.group(1)), float(match.group(2))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    for i in range(len(ring)):
        x1, y1 = ring[i][:2]
        x2, y2 = ring[(i + 1) % len(ring)][:2]
        if ((y1 > lat) != (y2 > lat)) and (
            lon < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-15) + x1
        ):
            inside = not inside
    return inside


def point_in_polygon(lon: float, lat: float, polygon: list[list[list[float]]]) -> bool:
    if not polygon or not point_in_ring(lon, lat, polygon[0]):
        return False
    return not any(point_in_ring(lon, lat, hole) for hole in polygon[1:])


def point_in_geojson_geometry(lon: float, lat: float, geometry: dict[str, Any]) -> bool:
    if not geometry:
        return False
    if geometry["type"] == "Polygon":
        return point_in_polygon(lon, lat, geometry["coordinates"])
    if geometry["type"] == "MultiPolygon":
        return any(point_in_polygon(lon, lat, poly) for poly in geometry["coordinates"])
    return False


def percentile_ranks(values: list[float], *, higher_is_better: bool = True) -> list[float]:
    if not values:
        return []
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return [100.0]
    result = []
    for value in values:
        lower = sum(1 for item in sorted_values if item < value)
        equal = sum(1 for item in sorted_values if item == value)
        percentile = 100.0 * (lower + 0.5 * equal) / n
        if not higher_is_better:
            percentile = 100.0 - percentile
        result.append(round(percentile, 1))
    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def log1p(value: float) -> float:
    return math.log1p(max(value, 0.0))


def expm1(value: float) -> float:
    return max(math.expm1(value), 0.0)
