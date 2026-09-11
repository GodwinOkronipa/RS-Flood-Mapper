"""
Realistic georeferenced remote sensing demonstration scenarios based on historic flood disasters:
  1. Indus River Basin Mega-Flood (Sindh, Pakistan - 2022)
  2. Houston / Buffalo Bayou Urban Storm Surge (Hurricane Harvey, Texas - 2017)
  3. Ahr Valley Flash Flood & Gorge Surge (Rhineland-Palatinate, Germany - 2021)

Includes real-world latitude/longitude bounding boxes and geotransforms to overlay onto OpenStreetMap.
"""

from typing import Dict, List, Tuple
import numpy as np


def generate_demo_scenario(scenario_type: str = "river") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate realistic multi-spectral satellite imagery and georeferencing metadata.

    Returns
    -------
    rgb_disp : np.ndarray
        uint8 RGB image (256, 256, 3)
    feature_stack : np.ndarray
        float32 multi-band stack (256, 256, 4): [R, G, B, NDWI]
    metadata : dict
        Includes geographic bounding box, center coordinates, and recommended route waypoints.
    """
    np.random.seed(101 if scenario_type == "river" else (202 if scenario_type == "coastal" else 303))
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]

    flood_mask = np.zeros((h, w), dtype=bool)

    if scenario_type == "river":
        scenario_name = "Indus River Basin Inundation (Sindh, Pakistan)"
        desc = "Catastrophic 2022 monsoon floodwaters overflowing the Indus River, inundating agrarian settlements and submerging secondary access routes."
        
        # Real-world coordinates (Dadu / Sehwan District, Sindh, Pakistan)
        bounds = [26.70, 67.95, 26.92, 68.22]  # [south_lat, west_lon, north_lat, east_lon]
        center = [26.81, 68.08]
        start_label = "Stranded Village (Dadu Sub-district)"
        goal_label = "UN High-Ground Relief Camp (Sehwan Ridge)"

        # Meandering river channel + flood expansion
        center_line = 128 + 42 * np.sin(x / 28.0) + 18 * np.cos(x / 14.0)
        dist_to_river = np.abs(y - center_line)
        flood_mask = dist_to_river < (30 + 14 * np.sin(x / 18.0))
        
        # Elevated arterial highway bridge (N-55 Indus Highway)
        bridge_mask = (np.abs(x - 128) < 9) & (dist_to_river < 45)
        flood_mask[bridge_mask] = False

        start_coord = (145, 25)
        goal_coord = (35, 220)

    elif scenario_type == "coastal":
        scenario_name = "Buffalo Bayou & Urban Surge (Houston, TX - Hurricane Harvey)"
        desc = "Extreme storm surge and reservoir release overtopping Buffalo Bayou, flooding urban neighborhoods and highway underpasses."
        
        # Real-world coordinates (Downtown Houston / Buffalo Bayou Park)
        bounds = [29.72, -95.44, 29.80, -95.32]
        center = [29.76, -95.38]
        start_label = "Inundated Residential Ward"
        goal_label = "Emergency Evacuation Center (George R. Brown)"

        # Bayou winding through urban grid
        bayou_line = 110 + 0.35 * x + 30 * np.sin(x / 32.0)
        flood_mask = (y > bayou_line - 15) & (y < bayou_line + 45)

        # Elevated freeway overpasses (I-45 / I-10 interchange)
        freeway_overpass = (np.abs(y - 120) < 6) & (x > 60) & (x < 200)
        flood_mask[freeway_overpass] = False

        start_coord = (185, 30)
        goal_coord = (40, 215)

    else:  # agricultural flash flood / European valley
        scenario_name = "Ahr Valley Flash Flood (Rhineland-Palatinate, Germany)"
        desc = "Severe 2021 European flash flooding following extreme rainfall, channeling down the steep river gorge and cutting bridge connections."
        
        # Real-world coordinates (Bad Neuenahr-Ahrweiler, Germany)
        bounds = [50.51, 7.04, 50.58, 7.18]
        center = [50.54, 7.11]
        start_label = "Gorge Settlement (Isolated Sector)"
        goal_label = "Hilltop Emergency Hospital (Grafschaft)"

        # Winding gorge valley
        gorge_line = 135 + 50 * np.sin(x / 25.0)
        dist_gorge = np.abs(y - gorge_line)
        flood_mask = dist_gorge < (24 + 10 * np.cos(x / 16.0))

        # Secondary mountain ridge road
        ridge_road = np.abs(y - 65) < 7
        flood_mask[ridge_road] = False

        start_coord = (165, 40)
        goal_coord = (35, 130)

    # Realistic Satellite Multi-spectral synthesis
    base_r = np.random.normal(88, 7, (h, w))
    base_g = np.random.normal(132, 10, (h, w))
    base_b = np.random.normal(68, 8, (h, w))

    # Arterial transport network (roads appear brighter gray in optical satellite)
    roads = (np.abs(y - 110) < 3) | (np.abs(x - 128) < 3)
    base_r[roads] = 165
    base_g[roads] = 165
    base_b[roads] = 165

    # Realistic turbid water body (Sentinel-2 B02/B03/B04 reflectance with sediment)
    turbidity = np.random.normal(0, 5, (h, w))
    base_r[flood_mask] = 32 + turbidity[flood_mask]
    base_g[flood_mask] = 68 + turbidity[flood_mask]
    base_b[flood_mask] = 118 + turbidity[flood_mask]

    rgb_disp = np.stack([
        np.clip(base_r, 0, 255).astype(np.uint8),
        np.clip(base_g, 0, 255).astype(np.uint8),
        np.clip(base_b, 0, 255).astype(np.uint8),
    ], axis=-1)

    norm_r = rgb_disp[:, :, 0].astype(np.float32) / 255.0
    norm_g = rgb_disp[:, :, 1].astype(np.float32) / 255.0
    norm_b = rgb_disp[:, :, 2].astype(np.float32) / 255.0
    
    ndwi = np.zeros((h, w), dtype=np.float32)
    ndwi[flood_mask] = np.random.uniform(0.38, 0.78, size=np.sum(flood_mask))
    ndwi[~flood_mask] = np.random.uniform(-0.55, -0.05, size=np.sum(~flood_mask))

    feature_stack = np.stack([norm_r, norm_g, norm_b, ndwi], axis=-1)

    metadata = {
        "scenario_name": scenario_name,
        "description": desc,
        "filename": f"sentinel_{scenario_type}_scene.tif",
        "dimensions": (h, w),
        "ground_truth_flood": flood_mask,
        "recommended_start": start_coord,
        "recommended_goal": goal_coord,
        "geo_bounds": bounds,  # [south, west, north, east]
        "geo_center": center,   # [lat, lon]
        "start_label": start_label,
        "goal_label": goal_label,
    }

    return rgb_disp, feature_stack, metadata


def pixel_to_latlon(
    row: int,
    col: int,
    height: int,
    width: int,
    geo_bounds: List[float]
) -> Tuple[float, float]:
    """
    Map image pixel (row, col) to real-world WGS84 (latitude, longitude).
    geo_bounds: [south_lat, west_lon, north_lat, east_lon]
    """
    s_lat, w_lon, n_lat, e_lon = geo_bounds
    # row 0 is north (top), row height is south (bottom)
    lat = n_lat - (row / height) * (n_lat - s_lat)
    # col 0 is west (left), col width is east (right)
    lon = w_lon + (col / width) * (e_lon - w_lon)
    return round(float(lat), 6), round(float(lon), 6)
