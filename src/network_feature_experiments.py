from __future__ import annotations

import heapq
import json
import math
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any

from build_features import HIGH_STREET_TYPES, load_osm_points
from feature_experiments import (
    GROUPS,
    LOG_FEATURES,
    evaluate_feature_set,
    nested_alpha_predictions,
)
from model_weekly_passers import metrics, rank_correlation
from utils import DATA_PROCESSED, DATA_RAW, OUTPUT_TABLES, haversine_m, log1p, median, read_csv, safe_float, write_csv


WFS_URL = "https://api.data.amsterdam.nl/v1/wfs/loopfietsnetwerk/v1"
NETWORK_RADIUS_M = 850.0
REGION_LINK_DISTANCE_M = 1_500.0
REGION_PADDING_M = 950.0

NETWORK_FEATURES = [
    "network_reach_400m",
    "pedestrian_route_centrality_800m",
    "destination_route_centrality_800m",
    "corridor_continuity_m",
    "local_walkway_width_m",
]

LOG_FEATURES.update({"network_reach_400m", "corridor_continuity_m", "local_walkway_width_m"})


def fetch_json(url: str, timeout: int = 180) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "ai-location-intelligence-ads/0.2"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def connected_location_regions(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    remaining = set(range(len(rows)))
    regions: list[list[dict[str, Any]]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            linked = {
                idx
                for idx in remaining
                if haversine_m(
                    safe_float(rows[current]["lat"]),
                    safe_float(rows[current]["lon"]),
                    safe_float(rows[idx]["lat"]),
                    safe_float(rows[idx]["lon"]),
                )
                <= REGION_LINK_DISTANCE_M
            }
            remaining -= linked
            component |= linked
            frontier.extend(linked)
        regions.append([rows[idx] for idx in sorted(component)])
    return sorted(regions, key=lambda region: min(row["location_name"] for row in region))


def region_bbox(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    lats = [safe_float(row["lat"]) for row in rows]
    lons = [safe_float(row["lon"]) for row in rows]
    center_lat = sum(lats) / len(lats)
    lat_pad = REGION_PADDING_M / 111_000.0
    lon_pad = REGION_PADDING_M / (111_000.0 * math.cos(math.radians(center_lat)))
    return min(lons) - lon_pad, min(lats) - lat_pad, max(lons) + lon_pad, max(lats) + lat_pad


def fetch_region_edges(region_index: int, bbox: tuple[float, float, float, float], refresh: bool) -> dict[str, Any]:
    path = DATA_RAW / f"loopfiets_edges_region_{region_index}.geojson"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))

    west, south, east, north = bbox
    page_size = 10_000
    start_index = 0
    features: list[dict[str, Any]] = []
    while True:
        query = urllib.parse.urlencode(
            {
                "service": "WFS",
                "version": "2.0.0",
                "request": "GetFeature",
                "typeNames": "app:edges",
                "outputFormat": "application/json",
                "srsName": "urn:ogc:def:crs:OGC::CRS84",
                "count": page_size,
                "startIndex": start_index,
                "bbox": f"{west},{south},{east},{north},urn:ogc:def:crs:OGC::CRS84",
            }
        )
        page = fetch_json(f"{WFS_URL}?{query}")
        page_features = page.get("features", [])
        features.extend(page_features)
        print(f"Region {region_index}: fetched {len(features):,} pedestrian edges", flush=True)
        if len(page_features) < page_size:
            break
        start_index += len(page_features)

    collection = {
        "type": "FeatureCollection",
        "source": WFS_URL,
        "bbox": [west, south, east, north],
        "features": features,
    }
    path.write_text(json.dumps(collection, separators=(",", ":")), encoding="utf-8")
    return collection


def feature_midpoint(feature: dict[str, Any]) -> tuple[float, float] | None:
    coordinates = feature.get("geometry", {}).get("coordinates", [])
    if len(coordinates) < 2:
        return None
    middle = coordinates[len(coordinates) // 2]
    return float(middle[1]), float(middle[0])


def local_graph(
    collection: dict[str, Any], lat: float, lon: float
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, tuple[float, float]], list[dict[str, Any]]]:
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    node_coordinates: dict[int, tuple[float, float]] = {}
    edges = []
    for feature in collection.get("features", []):
        midpoint = feature_midpoint(feature)
        if not midpoint or haversine_m(lat, lon, midpoint[0], midpoint[1]) > NETWORK_RADIUS_M + 150:
            continue
        props = feature.get("properties", {})
        if props.get("indicatie_voet_fiets") is False or props.get("indicatie_toegankelijkheid") is False:
            continue
        start = props.get("start_node_id")
        end = props.get("end_node_id")
        coordinates = feature.get("geometry", {}).get("coordinates", [])
        if start is None or end is None or len(coordinates) < 2:
            continue
        start = int(start)
        end = int(end)
        weight = safe_float(props.get("weight"))
        if weight <= 0:
            weight = haversine_m(coordinates[0][1], coordinates[0][0], coordinates[-1][1], coordinates[-1][0])
        node_coordinates[start] = (float(coordinates[0][1]), float(coordinates[0][0]))
        node_coordinates[end] = (float(coordinates[-1][1]), float(coordinates[-1][0]))
        adjacency[start].append((end, weight))
        adjacency[end].append((start, weight))
        edges.append(
            {
                "start": start,
                "end": end,
                "weight": weight,
                "mid_lat": midpoint[0],
                "mid_lon": midpoint[1],
                "width": safe_float(props.get("gewogen_gemiddelde_breedte"), -1.0),
            }
        )
    return adjacency, node_coordinates, edges


def nearest_node(lat: float, lon: float, coordinates: dict[int, tuple[float, float]]) -> int | None:
    if not coordinates:
        return None
    return min(coordinates, key=lambda node: haversine_m(lat, lon, coordinates[node][0], coordinates[node][1]))


def largest_component_nodes(adjacency: dict[int, list[tuple[int, float]]]) -> set[int]:
    unseen = set(adjacency)
    largest: set[int] = set()
    while unseen:
        seed = unseen.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            node = frontier.pop()
            neighbors = {neighbor for neighbor, _ in adjacency.get(node, [])}
            new_nodes = neighbors & unseen
            unseen -= new_nodes
            component |= new_nodes
            frontier.extend(new_nodes)
        if len(component) > len(largest):
            largest = component
    return largest


def dijkstra(
    adjacency: dict[int, list[tuple[int, float]]], start: int, limit: float | None = None
) -> dict[int, float]:
    distances = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances.get(node):
            continue
        if limit is not None and distance > limit:
            continue
        for neighbor, weight in adjacency.get(node, []):
            candidate = distance + weight
            if limit is not None and candidate > limit:
                continue
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def sample_boundary_nodes(
    lat: float, lon: float, coordinates: dict[int, tuple[float, float]], distances: dict[int, float]
) -> list[int]:
    bins: dict[int, tuple[float, int]] = {}
    for node, distance in distances.items():
        if distance < 500 or distance > 820:
            continue
        node_lat, node_lon = coordinates[node]
        x = (node_lon - lon) * math.cos(math.radians(lat))
        y = node_lat - lat
        angle = (math.atan2(y, x) + 2 * math.pi) % (2 * math.pi)
        sector = int(angle / (2 * math.pi) * 16) % 16
        if sector not in bins or distance > bins[sector][0]:
            bins[sector] = (distance, node)
    return [bins[key][1] for key in sorted(bins)]


def route_through_share(
    adjacency: dict[int, list[tuple[int, float]]],
    candidate: int,
    candidate_distances: dict[int, float],
    origins: list[int],
    destinations: list[int],
    tolerance: float = 1.08,
) -> float:
    checked = 0
    through = 0
    destination_set = set(destinations)
    for origin in dict.fromkeys(origins):
        distances = dijkstra(adjacency, origin, limit=2_200)
        for destination in destination_set:
            if destination == origin or destination not in distances or destination not in candidate_distances:
                continue
            direct = distances[destination]
            if direct < 150:
                continue
            via_candidate = distances.get(candidate, float("inf")) + candidate_distances[destination]
            checked += 1
            if via_candidate <= direct * tolerance + 20:
                through += 1
    return through / checked if checked else 0.0


def select_nearby_points(
    points: list[dict[str, Any]], lat: float, lon: float, radius: float, maximum: int
) -> list[dict[str, Any]]:
    nearby = [
        (haversine_m(lat, lon, safe_float(point["lat"]), safe_float(point["lon"])), point)
        for point in points
    ]
    nearby = [item for item in nearby if item[0] <= radius]
    nearby.sort(key=lambda item: item[0])
    if len(nearby) <= maximum:
        return [item[1] for item in nearby]
    step = (len(nearby) - 1) / (maximum - 1)
    return [nearby[round(index * step)][1] for index in range(maximum)]


def corridor_continuity(
    candidate: int, adjacency: dict[int, list[tuple[int, float]]], maximum: float = 1_200
) -> float:
    def follow(previous: int, current: int, distance: float) -> float:
        seen = {previous, current}
        while distance < maximum:
            choices = [(node, weight) for node, weight in adjacency.get(current, []) if node != previous]
            if len(adjacency.get(current, [])) != 2 or len(choices) != 1:
                break
            next_node, weight = choices[0]
            if next_node in seen:
                break
            distance += weight
            previous, current = current, next_node
            seen.add(current)
        return distance

    branches = adjacency.get(candidate, [])
    if not branches:
        return 0.0
    lengths = sorted((follow(candidate, neighbor, weight) for neighbor, weight in branches), reverse=True)
    return sum(lengths[:2])


def compute_network_features(
    row: dict[str, Any], collection: dict[str, Any], osm_points: list[dict[str, Any]]
) -> dict[str, Any]:
    lat = safe_float(row["lat"])
    lon = safe_float(row["lon"])
    adjacency, coordinates, edges = local_graph(collection, lat, lon)
    main_component = largest_component_nodes(adjacency)
    main_coordinates = {node: coordinates[node] for node in main_component if node in coordinates}
    candidate = nearest_node(lat, lon, main_coordinates)
    if candidate is None:
        return {"location_name": row["location_name"], **{name: 0.0 for name in NETWORK_FEATURES}}

    candidate_distances = dijkstra(adjacency, candidate, limit=2_200)
    reachable_coordinates = {
        node: coordinates[node] for node in candidate_distances if node in coordinates
    }
    reach_nodes = {node for node, distance in candidate_distances.items() if distance <= 400}
    reach_length = sum(
        edge["weight"]
        for edge in edges
        if edge["start"] in reach_nodes or edge["end"] in reach_nodes
    )

    boundary = sample_boundary_nodes(lat, lon, coordinates, candidate_distances)
    pedestrian_centrality = route_through_share(
        adjacency, candidate, candidate_distances, boundary, boundary
    )

    transit_points = [point for point in osm_points if point.get("category") == "transit"]
    destination_points = [
        point
        for point in osm_points
        if point.get("category") == "tourism"
        or (point.get("category") == "shop" and point.get("shop_type") in HIGH_STREET_TYPES)
    ]
    local_origins = select_nearby_points(transit_points, lat, lon, NETWORK_RADIUS_M, 12)
    local_destinations = select_nearby_points(destination_points, lat, lon, NETWORK_RADIUS_M, 24)
    origin_nodes = [
        nearest_node(safe_float(point["lat"]), safe_float(point["lon"]), reachable_coordinates)
        for point in local_origins
    ]
    destination_nodes = [
        nearest_node(safe_float(point["lat"]), safe_float(point["lon"]), reachable_coordinates)
        for point in local_destinations
    ]
    destination_centrality = route_through_share(
        adjacency,
        candidate,
        candidate_distances,
        [node for node in origin_nodes if node is not None],
        [node for node in destination_nodes if node is not None],
    )

    local_widths = [
        edge["width"]
        for edge in edges
        if edge["width"] > 0 and haversine_m(lat, lon, edge["mid_lat"], edge["mid_lon"]) <= 60
    ]
    node_lat, node_lon = coordinates[candidate]
    return {
        "location_name": row["location_name"],
        "network_reach_400m": round(reach_length, 1),
        "pedestrian_route_centrality_800m": round(pedestrian_centrality, 4),
        "destination_route_centrality_800m": round(destination_centrality, 4),
        "corridor_continuity_m": round(corridor_continuity(candidate, adjacency), 1),
        "local_walkway_width_m": round(median(local_widths), 2) if local_widths else 0.0,
        "network_snap_distance_m": round(haversine_m(lat, lon, node_lat, node_lon), 1),
        "network_nodes_850m": sum(distance <= NETWORK_RADIUS_M for distance in candidate_distances.values()),
        "network_boundary_samples": len(boundary),
    }


def experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    production = GROUPS["compact_market_context"]
    candidates = {
        "production_current": production,
        **{f"production_plus__{feature}": production + [feature] for feature in NETWORK_FEATURES},
        "production_plus__network_centrality": production
        + ["pedestrian_route_centrality_800m", "destination_route_centrality_800m"],
        "production_plus__network_access": production
        + ["network_reach_400m", "corridor_continuity_m", "local_walkway_width_m"],
        "production_plus__all_network": production + NETWORK_FEATURES,
    }
    y = [log1p(safe_float(row["weekly_passers"])) for row in rows]
    results = []
    for name, features in candidates.items():
        standard = evaluate_feature_set(rows, name, features)
        nested_predictions, alphas = nested_alpha_predictions(rows, features)
        nested = metrics(y, nested_predictions)
        result = {
            **standard,
            "nested_rmse_log": round(nested["rmse_log"], 4),
            "nested_mae": round(nested["mae"]),
            "nested_mape_pct": round(nested["mape_pct"], 1),
            "nested_rank_correlation": round(rank_correlation(y, nested_predictions), 3),
            "nested_alphas": ";".join(str(alpha) for alpha in alphas),
        }
        results.append(result)
    baseline = next(row for row in results if row["feature_set"] == "production_current")
    for row in results:
        row["nested_rmse_improvement_pct"] = round(
            100 * (safe_float(baseline["nested_rmse_log"]) - safe_float(row["nested_rmse_log"]))
            / safe_float(baseline["nested_rmse_log"]),
            2,
        )
    return sorted(results, key=lambda row: (row["nested_rmse_log"], row["nested_mae"]))


def main() -> None:
    refresh = "--refresh-network" in sys.argv
    rows = [row for row in read_csv(DATA_PROCESSED / "site_features_weekly.csv") if safe_float(row.get("weekly_passers")) > 0]
    osm_points = load_osm_points()
    regions = connected_location_regions(rows)
    print(f"Using {len(regions)} bounded pedestrian-network regions", flush=True)
    collections = [fetch_region_edges(index + 1, region_bbox(region), refresh) for index, region in enumerate(regions)]

    location_to_region = {
        row["location_name"]: index
        for index, region in enumerate(regions)
        for row in region
    }
    feature_rows = []
    for index, row in enumerate(rows, start=1):
        region_index = location_to_region[row["location_name"]]
        features = compute_network_features(row, collections[region_index], osm_points)
        feature_rows.append(features)
        row.update(features)
        print(f"Computed network features {index}/{len(rows)}: {row['location_name']}", flush=True)

    write_csv(DATA_PROCESSED / "site_network_features.csv", feature_rows)
    results = experiment(rows)
    write_csv(OUTPUT_TABLES / "network_feature_experiment_results.csv", results)
    print("Network feature experiment complete")
    for row in results:
        print(
            f"{row['feature_set']:<55} nested_rmse={row['nested_rmse_log']:.3f} "
            f"nested_mae={row['nested_mae']:.0f} improvement={row['nested_rmse_improvement_pct']:.1f}%"
        )


if __name__ == "__main__":
    main()
