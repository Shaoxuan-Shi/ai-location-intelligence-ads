from __future__ import annotations

import csv
import json
import sys
import urllib.parse
from collections import defaultdict
from datetime import date

from utils import (
    DATA_PROCESSED,
    DATA_RAW,
    ensure_dirs,
    fetch_text_cached,
    median,
    parse_rd_point,
    quantile,
    rd_to_wgs,
    safe_float,
    safe_int,
    write_csv,
)


AMS_API = "https://api.data.amsterdam.nl/v1"
OVERPASS_API = "https://overpass-api.de/api/interpreter"


def fetch_crowdmonitor_weekly(refresh: bool = False) -> list[dict[str, str]]:
    url = f"{AMS_API}/crowdmonitor/passanten?periode=week&_format=csv&_pageSize=500&_sort=-datumUur"
    text = fetch_text_cached(url, DATA_RAW / "crowdmonitor_weekly_sorted.csv", refresh=refresh)
    all_rows = list(csv.DictReader(text.splitlines()))

    if not all_rows:
        raise RuntimeError("No Crowdmonitor weekly rows fetched.")

    latest_date = max(row.get("datumUur", "") for row in all_rows)
    latest_rows = [row for row in all_rows if row.get("datumUur") == latest_date]

    geometry_lookup = {row["naamLocatie"]: row.get("geometrie", "") for row in all_rows if row.get("geometrie")}

    output_rows: list[dict[str, str]] = []
    for row in latest_rows:
        geometry = row.get("geometrie") or geometry_lookup.get(row.get("naamLocatie", ""), "")
        if not geometry:
            continue
        x, y = parse_rd_point(geometry)
        lat, lon = rd_to_wgs(x, y)
        output_rows.append(
            {
                "location_name": row.get("naamLocatie", ""),
                "sensor": row.get("sensor", ""),
                "date": row.get("datumUur", ""),
                "weekly_passers": row.get("aantalPassanten", ""),
                "rd_x": round(x, 3),
                "rd_y": round(y, 3),
                "lat": round(lat, 7),
                "lon": round(lon, 7),
            }
        )

    output_rows.sort(key=lambda row: int(float(row["weekly_passers"])), reverse=True)
    write_csv(DATA_RAW / "crowdmonitor_weekly_latest_locations.csv", output_rows)
    return output_rows


def month_starts(start_year: int, end_year: int) -> list[date]:
    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            months.append(date(year, month, 1))
    return months


def next_month(day: date) -> date:
    if day.month == 12:
        return date(day.year + 1, 1, 1)
    return date(day.year, day.month + 1, 1)


def fetch_crowdmonitor_weekly_history(refresh: bool = False) -> list[dict[str, str]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    for start in month_starts(2023, 2024):
        end = next_month(start)
        url = (
            f"{AMS_API}/crowdmonitor/passanten"
            f"?periode=week&datumUur%5Bgte%5D={start.isoformat()}&datumUur%5Blt%5D={end.isoformat()}"
            "&_format=csv&_pageSize=1000&_sort=datumUur"
        )
        filename = f"crowdmonitor_weekly_{start.year}_{start.month:02d}.csv"
        text = fetch_text_cached(url, DATA_RAW / filename, refresh=refresh)
        for row in csv.DictReader(text.splitlines()):
            if row.get("periode") == "week":
                rows_by_id[row.get("id") or f"{row.get('naamLocatie')}-{row.get('datumUur')}"] = row

    history_rows = list(rows_by_id.values())
    history_rows.sort(key=lambda row: (row.get("datumUur", ""), row.get("naamLocatie", "")))
    write_csv(DATA_RAW / "crowdmonitor_weekly_history.csv", history_rows)
    write_weekly_location_summary(history_rows)
    return history_rows


def write_weekly_location_summary(history_rows: list[dict[str, str]]) -> None:
    latest_locations = fetch_crowdmonitor_weekly(refresh=False)
    coord_lookup = {
        row["location_name"]: {
            "sensor": row.get("sensor", ""),
            "lat": row.get("lat", ""),
            "lon": row.get("lon", ""),
            "latest_week_date": row.get("date", ""),
            "latest_weekly_passers": row.get("weekly_passers", ""),
        }
        for row in latest_locations
    }

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in history_rows:
        if row.get("naamLocatie") in coord_lookup and safe_int(row.get("aantalPassanten")) > 0:
            grouped[row["naamLocatie"]].append(row)

    history_by_location: dict[str, dict[str, float]] = {}
    for name, rows in grouped.items():
        history_by_location[name] = {
            row["datumUur"]: safe_float(row["aantalPassanten"])
            for row in rows
        }

    location_medians = {
        name: median(list(values.values()))
        for name, values in history_by_location.items()
    }
    ratios_by_week: dict[str, list[float]] = defaultdict(list)
    for name, values in history_by_location.items():
        for week, count in values.items():
            ratios_by_week[week].append(count / max(location_medians[name], 1.0))
    week_factors = {week: median(ratios) for week, ratios in ratios_by_week.items()}

    coordinate_groups: dict[tuple[float, float], list[str]] = defaultdict(list)
    for name in grouped:
        coord = coord_lookup[name]
        coordinate_groups[(round(safe_float(coord["lat"]), 6), round(safe_float(coord["lon"]), 6))].append(name)

    summary_rows = []
    for names in coordinate_groups.values():
        common_dates = sorted(set.intersection(*(set(history_by_location[name]) for name in names)))
        raw_counts = [
            sum(history_by_location[name][week] for name in names) / len(names)
            for week in common_dates
        ]
        raw_median = median(raw_counts)
        retained = []
        excluded_week_count = 0
        for week, count in zip(common_dates, raw_counts):
            sensor_failure = count < 0.20 * raw_median or count > 5.0 * raw_median
            incomplete_week = week_factors.get(week, 1.0) < 0.60
            if sensor_failure or incomplete_week:
                excluded_week_count += 1
            else:
                retained.append(count)
        counts = retained or raw_counts
        med = median(counts)
        mean_value = sum(counts) / len(counts)
        p25 = quantile(counts, 0.25)
        p75 = quantile(counts, 0.75)
        coord = coord_lookup[names[0]]
        if len(names) == 1:
            display_name = names[0]
        else:
            common_prefix = names[0].split(" (")[0]
            display_name = f"{common_prefix} (direction-neutral)"
        summary_rows.append(
            {
                "location_name": display_name,
                "source_location_names": " | ".join(names),
                "sensor": " | ".join(coord_lookup[name]["sensor"] for name in names),
                "source_sensor_count": len(names),
                "lat": coord["lat"],
                "lon": coord["lon"],
                "weekly_observation_count": len(counts),
                "excluded_anomalous_week_count": excluded_week_count,
                "first_week": common_dates[0],
                "last_week": common_dates[-1],
                "target_definition": "direction_neutral_normal_week_median",
                "raw_median_weekly_passers": round(raw_median),
                "median_weekly_passers": round(med),
                "mean_weekly_passers": round(mean_value),
                "p25_weekly_passers": round(p25),
                "p75_weekly_passers": round(p75),
                "min_weekly_passers": round(min(counts)),
                "max_weekly_passers": round(max(counts)),
                "iqr_weekly_passers": round(p75 - p25),
                "variability_ratio": round((p75 - p25) / max(med, 1.0), 3),
                "latest_week_date": coord["latest_week_date"],
                "latest_weekly_passers": round(
                    sum(safe_float(coord_lookup[name]["latest_weekly_passers"]) for name in names) / len(names)
                ),
            }
        )

    summary_rows.sort(key=lambda row: safe_float(row["median_weekly_passers"]), reverse=True)
    write_csv(DATA_PROCESSED / "crowdmonitor_weekly_location_summary.csv", summary_rows)

    all_dates = sorted({row.get("datumUur", "") for row in history_rows})
    audit_rows = [
        {"metric": "history_rows", "value": len(history_rows)},
        {"metric": "history_locations", "value": len({row.get("naamLocatie", "") for row in history_rows})},
        {"metric": "usable_locations_with_coordinates", "value": len(summary_rows)},
        {"metric": "first_week", "value": all_dates[0] if all_dates else ""},
        {"metric": "last_week", "value": all_dates[-1] if all_dates else ""},
        {"metric": "weekly_dates", "value": len(all_dates)},
    ]
    write_csv(DATA_PROCESSED / "crowdmonitor_weekly_audit.csv", audit_rows)


def fetch_amsterdam_geojson(refresh: bool = False) -> None:
    sources = {
        "winkelgebieden.geojson": f"{AMS_API}/winkelgebieden/winkelgebieden?_format=geojson&_pageSize=1000",
        "horeca.geojson": f"{AMS_API}/horeca/exploitatievergunning?_format=geojson&_pageSize=10000",
    }
    for filename, url in sources.items():
        fetch_text_cached(url, DATA_RAW / filename, refresh=refresh)


def fetch_osm_pois(refresh: bool = False) -> None:
    locations = list(csv.DictReader((DATA_RAW / "crowdmonitor_weekly_latest_locations.csv").read_text(encoding="utf-8").splitlines()))
    lats = [safe_float(row["lat"]) for row in locations]
    lons = [safe_float(row["lon"]) for row in locations]
    margin = 0.012
    south, west, north, east = min(lats) - margin, min(lons) - margin, max(lats) + margin, max(lons) + margin
    query = f"""
    [out:json][timeout:90];
    (
      node["shop"]({south},{west},{north},{east});
      node["tourism"]({south},{west},{north},{east});
      node["historic"]({south},{west},{north},{east});
      node["public_transport"]({south},{west},{north},{east});
      node["highway"="bus_stop"]({south},{west},{north},{east});
      node["railway"~"station|halt|tram_stop|subway_entrance"]({south},{west},{north},{east});
    );
    out body;
    """
    url = OVERPASS_API + "?data=" + urllib.parse.quote(query)
    text = fetch_text_cached(url, DATA_RAW / "osm_pois_amsterdam.json", refresh=refresh)
    json.loads(text)


def fetch_osm_streets(refresh: bool = False) -> None:
    locations = list(csv.DictReader((DATA_RAW / "crowdmonitor_weekly_latest_locations.csv").read_text(encoding="utf-8").splitlines()))
    lats = [safe_float(row["lat"]) for row in locations]
    lons = [safe_float(row["lon"]) for row in locations]
    margin = 0.012
    south, west, north, east = min(lats) - margin, min(lons) - margin, max(lats) + margin, max(lons) + margin
    query = f"""
    [out:json][timeout:120];
    way["highway"~"^(pedestrian|footway|path|living_street|residential|tertiary|secondary|primary)$"]
      ({south},{west},{north},{east});
    out body geom;
    """
    url = OVERPASS_API + "?data=" + urllib.parse.quote(query)
    text = fetch_text_cached(url, DATA_RAW / "osm_streets_amsterdam.json", refresh=refresh)
    json.loads(text)


def main() -> None:
    ensure_dirs()
    refresh = "--refresh" in sys.argv
    refresh_osm = refresh or "--refresh-osm" in sys.argv
    refresh_streets = refresh or "--refresh-streets" in sys.argv
    refresh_history = refresh or "--refresh-history" in sys.argv
    fetch_crowdmonitor_weekly(refresh=refresh)
    fetch_crowdmonitor_weekly_history(refresh=refresh_history)
    fetch_amsterdam_geojson(refresh=refresh)
    fetch_osm_pois(refresh=refresh_osm)
    fetch_osm_streets(refresh=refresh_streets)
    print("Fetched raw data into data/raw")


if __name__ == "__main__":
    main()
