from __future__ import annotations

from collections import Counter
import math
from typing import Any

from utils import (
    DATA_PROCESSED,
    DATA_RAW,
    haversine_m,
    point_in_geojson_geometry,
    percentile_ranks,
    read_csv,
    read_json,
    safe_float,
    safe_int,
    write_csv,
)


HIGH_STREET_TYPES = {"clothes", "shoes", "jewelry", "department_store", "gift", "cosmetics", "fashion_accessories"}
DESTINATION_TYPES = {"attraction", "museum", "gallery", "viewpoint", "zoo", "aquarium", "theme_park"}
LODGING_TYPES = {"hotel", "hostel", "guest_house", "apartment", "motel"}
HIGH_CAPACITY_TRANSIT_TYPES = {"station", "halt", "subway_entrance"}
NIGHTLIFE_CATEGORIES = {"Nachtzaak", "Café", "Café met zaalverhuur", "Sociëteit"}


def element_lon_lat(element: dict[str, Any]) -> tuple[float, float] | None:
    if "lat" in element and "lon" in element:
        return float(element["lon"]), float(element["lat"])
    center = element.get("center")
    if center and "lat" in center and "lon" in center:
        return float(center["lon"]), float(center["lat"])
    return None


def osm_category(tags: dict[str, Any]) -> str | None:
    if tags.get("shop"):
        return "shop"
    if tags.get("tourism") or tags.get("historic"):
        return "tourism"
    if tags.get("public_transport"):
        return "transit"
    if tags.get("highway") == "bus_stop":
        return "transit"
    if tags.get("railway") in {"station", "halt", "tram_stop", "subway_entrance"}:
        return "transit"
    return None


def load_osm_points() -> list[dict[str, Any]]:
    data = read_json(DATA_RAW / "osm_pois_amsterdam.json")
    points = []
    seen = set()
    for element in data.get("elements", []):
        lon_lat = element_lon_lat(element)
        tags = element.get("tags", {})
        category = osm_category(tags)
        if not lon_lat or not category:
            continue
        key = (category, round(lon_lat[0], 6), round(lon_lat[1], 6), tags.get("name", ""))
        if key in seen:
            continue
        seen.add(key)
        points.append(
            {
                "category": category,
                "lon": lon_lat[0],
                "lat": lon_lat[1],
                "name": tags.get("name", ""),
                "raw_type": tags.get("shop") or tags.get("tourism") or tags.get("public_transport") or tags.get("railway") or "",
                "shop_type": tags.get("shop", ""),
                "tourism_type": tags.get("tourism", ""),
                "historic_type": tags.get("historic", ""),
                "transit_type": tags.get("railway") or tags.get("public_transport") or tags.get("highway", ""),
            }
        )
    return points


def load_osm_streets() -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    path = DATA_RAW / "osm_streets_amsterdam.json"
    if not path.exists():
        return [], []
    data = read_json(path)
    streets = []
    node_way_count: Counter[int] = Counter()
    node_coordinates: dict[int, tuple[float, float]] = {}
    for element in data.get("elements", []):
        geometry = element.get("geometry", [])
        node_ids = element.get("nodes", [])
        if len(geometry) < 2:
            continue
        coords = [(float(point["lat"]), float(point["lon"])) for point in geometry]
        streets.append({"highway": element.get("tags", {}).get("highway", ""), "coordinates": coords})
        for node_id, point in zip(node_ids, geometry):
            node_way_count[int(node_id)] += 1
            node_coordinates[int(node_id)] = (float(point["lat"]), float(point["lon"]))
    intersections = [
        {"lat": node_coordinates[node_id][0], "lon": node_coordinates[node_id][1], "ways": float(count)}
        for node_id, count in node_way_count.items()
        if count >= 3 and node_id in node_coordinates
    ]
    return streets, intersections


def load_horeca_points() -> list[dict[str, Any]]:
    data = read_json(DATA_RAW / "horeca.geojson")
    points = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        if not geometry or geometry.get("type") != "Point":
            continue
        lon, lat = geometry["coordinates"][:2]
        props = feature.get("properties", {})
        points.append(
            {
                "lon": float(lon),
                "lat": float(lat),
                "category": props.get("zaakCategorie") or "",
                "specificatie": props.get("zaakSpecificatie") or "",
            }
        )
    return points


def shopping_area_match(lon: float, lat: float, winkelgebieden: dict[str, Any]) -> tuple[int, str, str]:
    for feature in winkelgebieden.get("features", []):
        if point_in_geojson_geometry(lon, lat, feature.get("geometry", {})):
            props = feature.get("properties", {})
            return 1, props.get("gebiedsnaam") or "", props.get("categorie") or ""
    return 0, "", ""


def count_nearby(points: list[dict[str, Any]], lat: float, lon: float, radius_m: float, category: str | None = None) -> int:
    total = 0
    for point in points:
        if category and point.get("category") != category:
            continue
        if haversine_m(lat, lon, float(point["lat"]), float(point["lon"])) <= radius_m:
            total += 1
    return total


def nearest_distance(points: list[dict[str, Any]], lat: float, lon: float, category: str | None = None) -> float:
    distances = []
    for point in points:
        if category and point.get("category") != category:
            continue
        distances.append(haversine_m(lat, lon, float(point["lat"]), float(point["lon"])))
    return min(distances) if distances else 9999.0


def count_matching(points: list[dict[str, Any]], lat: float, lon: float, radius_m: float, predicate) -> int:
    return sum(
        1
        for point in points
        if predicate(point) and haversine_m(lat, lon, float(point["lat"]), float(point["lon"])) <= radius_m
    )


def street_lengths_nearby(streets: list[dict[str, Any]], lat: float, lon: float, radius_m: float) -> tuple[float, float]:
    total = 0.0
    pedestrian = 0.0
    pedestrian_types = {"pedestrian", "footway", "path", "living_street"}
    for street in streets:
        coords = street["coordinates"]
        for start, end in zip(coords, coords[1:]):
            midpoint_lat = (start[0] + end[0]) / 2
            midpoint_lon = (start[1] + end[1]) / 2
            if haversine_m(lat, lon, midpoint_lat, midpoint_lon) > radius_m:
                continue
            length = haversine_m(start[0], start[1], end[0], end[1])
            total += length
            if street["highway"] in pedestrian_types:
                pedestrian += length
    return total, pedestrian


def add_percentiles(rows: list[dict[str, Any]]) -> None:
    percentile_specs = [
        ("shop_count_80m", True),
        ("shop_count_120m", True),
        ("horeca_count_120m", True),
        ("transit_stop_count_250m", True),
        ("distance_to_nearest_transit_m", False),
        ("tourism_poi_count_300m", True),
        ("distance_to_amsterdam_center_m", False),
        ("pedestrian_street_length_200m", True),
        ("intersection_count_150m", True),
        ("weekly_passers", True),
    ]
    for column, higher_is_better in percentile_specs:
        values = [safe_float(row.get(column)) for row in rows]
        percentiles = percentile_ranks(values, higher_is_better=higher_is_better)
        for row, percentile in zip(rows, percentiles):
            row[f"{column}_percentile"] = percentile


def main() -> None:
    summary_path = DATA_PROCESSED / "crowdmonitor_weekly_location_summary.csv"
    locations = read_csv(summary_path if summary_path.exists() else DATA_RAW / "crowdmonitor_weekly_latest_locations.csv")
    winkelgebieden = read_json(DATA_RAW / "winkelgebieden.geojson")
    horeca_points = load_horeca_points()
    osm_points = load_osm_points()
    streets, intersections = load_osm_streets()

    shop_points = [point for point in osm_points if point["category"] == "shop"]
    transit_points = [point for point in osm_points if point["category"] == "transit"]
    tourism_points = [point for point in osm_points if point["category"] == "tourism"]

    output_rows: list[dict[str, Any]] = []
    for location in locations:
        lat = safe_float(location["lat"])
        lon = safe_float(location["lon"])
        in_area, area_name, area_category = shopping_area_match(lon, lat, winkelgebieden)
        street_length_200m, pedestrian_length_200m = street_lengths_nearby(streets, lat, lon, 200)
        row: dict[str, Any] = {
            "location_name": location["location_name"],
            "source_location_names": location.get("source_location_names", location["location_name"]),
            "sensor": location["sensor"],
            "first_week": location.get("first_week", location.get("date", "")),
            "last_week": location.get("last_week", location.get("date", "")),
            "weekly_observation_count": safe_int(location.get("weekly_observation_count", 1)),
            "weekly_passers": safe_int(location.get("median_weekly_passers", location.get("weekly_passers", 0))),
            "target_definition": location.get(
                "target_definition",
                "median_weekly_passers" if "median_weekly_passers" in location else "latest_weekly_passers",
            ),
            "source_sensor_count": safe_int(location.get("source_sensor_count", 1)),
            "excluded_anomalous_week_count": safe_int(location.get("excluded_anomalous_week_count", 0)),
            "raw_median_weekly_passers": safe_int(
                location.get("raw_median_weekly_passers", location.get("median_weekly_passers", location.get("weekly_passers", 0)))
            ),
            "mean_weekly_passers": safe_int(location.get("mean_weekly_passers", location.get("weekly_passers", 0))),
            "p25_weekly_passers": safe_int(location.get("p25_weekly_passers", location.get("weekly_passers", 0))),
            "p75_weekly_passers": safe_int(location.get("p75_weekly_passers", location.get("weekly_passers", 0))),
            "iqr_weekly_passers": safe_int(location.get("iqr_weekly_passers", 0)),
            "variability_ratio": safe_float(location.get("variability_ratio", 0.0)),
            "latest_weekly_passers": safe_int(location.get("latest_weekly_passers", location.get("weekly_passers", 0))),
            "lat": lat,
            "lon": lon,
            "shop_count_80m": count_nearby(osm_points, lat, lon, 80, "shop"),
            "shop_count_120m": count_nearby(osm_points, lat, lon, 120, "shop"),
            "shop_count_40m": count_nearby(osm_points, lat, lon, 40, "shop"),
            "shop_count_200m": count_nearby(osm_points, lat, lon, 200, "shop"),
            "shop_count_300m": count_nearby(osm_points, lat, lon, 300, "shop"),
            "distance_to_nearest_shop_m": round(nearest_distance(shop_points, lat, lon), 1),
            "high_street_shop_count_120m": count_matching(
                shop_points, lat, lon, 120, lambda point: point.get("shop_type") in HIGH_STREET_TYPES
            ),
            "horeca_count_120m": count_nearby(horeca_points, lat, lon, 120),
            "horeca_count_60m": count_nearby(horeca_points, lat, lon, 60),
            "horeca_count_200m": count_nearby(horeca_points, lat, lon, 200),
            "horeca_count_300m": count_nearby(horeca_points, lat, lon, 300),
            "distance_to_nearest_horeca_m": round(nearest_distance(horeca_points, lat, lon), 1),
            "nightlife_count_300m": count_matching(
                horeca_points, lat, lon, 300, lambda point: point.get("category") in NIGHTLIFE_CATEGORIES
            ),
            "transit_stop_count_250m": count_nearby(osm_points, lat, lon, 250, "transit"),
            "transit_stop_count_100m": count_nearby(osm_points, lat, lon, 100, "transit"),
            "transit_stop_count_500m": count_nearby(osm_points, lat, lon, 500, "transit"),
            "distance_to_nearest_transit_m": round(nearest_distance(osm_points, lat, lon, "transit"), 1),
            "high_capacity_transit_count_500m": count_matching(
                transit_points, lat, lon, 500, lambda point: point.get("transit_type") in HIGH_CAPACITY_TRANSIT_TYPES
            ),
            "tourism_poi_count_300m": count_nearby(osm_points, lat, lon, 300, "tourism"),
            "tourism_poi_count_100m": count_nearby(osm_points, lat, lon, 100, "tourism"),
            "tourism_poi_count_500m": count_nearby(osm_points, lat, lon, 500, "tourism"),
            "distance_to_nearest_tourism_poi_m": round(nearest_distance(tourism_points, lat, lon), 1),
            "visitor_destination_count_500m": count_matching(
                tourism_points,
                lat,
                lon,
                500,
                lambda point: point.get("tourism_type") in DESTINATION_TYPES or bool(point.get("historic_type")),
            ),
            "lodging_count_500m": count_matching(
                tourism_points, lat, lon, 500, lambda point: point.get("tourism_type") in LODGING_TYPES
            ),
            "in_official_shopping_area": in_area,
            "in_key_shopping_area": int(area_category == "K"),
            "shopping_area_name": area_name,
            "shopping_area_category": area_category,
            "distance_to_amsterdam_center_m": round(haversine_m(lat, lon, 52.3731, 4.8922), 1),
            "walkable_street_length_200m": round(street_length_200m, 1),
            "pedestrian_street_length_200m": round(pedestrian_length_200m, 1),
            "pedestrian_street_share_200m": round(pedestrian_length_200m / max(street_length_200m, 1.0), 3),
            "intersection_count_150m": count_nearby(intersections, lat, lon, 150),
            "distance_to_nearest_intersection_m": round(nearest_distance(intersections, lat, lon), 1),
        }
        row["retail_concentration_80_300"] = round(
            row["shop_count_80m"] / max(row["shop_count_300m"], 1), 3
        )
        row["horeca_concentration_120_300"] = round(
            row["horeca_count_120m"] / max(row["horeca_count_300m"], 1), 3
        )
        row["active_poi_categories_300m"] = sum(
            value > 0
            for value in [
                row["shop_count_300m"],
                row["horeca_count_300m"],
                row["transit_stop_count_250m"],
                row["tourism_poi_count_300m"],
            ]
        )
        output_rows.append(row)

    add_percentiles(output_rows)
    output_rows.sort(key=lambda row: row["weekly_passers"], reverse=True)
    write_csv(DATA_PROCESSED / "site_features_weekly.csv", output_rows)

    osm_counts = Counter(point["category"] for point in osm_points)
    summary = [
        {"metric": "candidate_locations", "value": len(output_rows)},
        {"metric": "horeca_points", "value": len(horeca_points)},
        {"metric": "osm_shop_points", "value": osm_counts.get("shop", 0)},
        {"metric": "osm_transit_points", "value": osm_counts.get("transit", 0)},
        {"metric": "osm_tourism_points", "value": osm_counts.get("tourism", 0)},
    ]
    write_csv(DATA_PROCESSED / "data_summary.csv", summary)
    print(f"Built features for {len(output_rows)} locations")


if __name__ == "__main__":
    main()
