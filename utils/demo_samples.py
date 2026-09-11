"""
Georeferenced flood scenarios focused on Ghana and West Africa:
  1. Akosombo Dam Spillage (Volta Region, Ghana - October 2023)
  2. White Volta & Bagre Dam Basin Overflow (Northern Region, Ghana)
  3. Greater Accra Flash Flood (Odaw River & Kwame Nkrumah Circle, Ghana)

Provides real GPS coordinates and bounding boxes for OpenStreetMap overlays.
"""

from typing import Dict, List, Tuple
import numpy as np


def generate_demo_scenario(scenario_type: str = "volta") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic multi-spectral imagery matching Ghana flood contexts.

    Returns
    -------
    rgb_disp : np.ndarray
        uint8 RGB image (256, 256, 3)
    feature_stack : np.ndarray
        float32 multi-band stack (256, 256, 4): [R, G, B, NDWI]
    metadata : dict
        Geographic bounding box, center coordinates, and waypoints.
    """
    np.random.seed(101 if scenario_type == "volta" else (202 if scenario_type == "white_volta" else 303))
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]

    flood_mask = np.zeros((h, w), dtype=bool)

    if scenario_type == "volta":
        scenario_name = "Akosombo Dam Spillage (Volta Region, Ghana - 2023)"
        desc = "Controlled spillage from Akosombo Dam overflowing the Lower Volta River basin, inundating Mepe, Battor, and surrounding agrarian communities."
        
        # Lower Volta basin near Mepe / Battor / Sogakope
        bounds = [5.92, 0.50, 6.08, 0.68]  # [south, west, north, east]
        center = [6.00, 0.59]
        start_label = "Mepe Submerged Community"
        goal_label = "Sogakope Elevated Relief Center"

        # Winding Lower Volta River channel
        center_line = 130 + 40 * np.sin(x / 30.0) + 16 * np.cos(x / 16.0)
        dist_to_river = np.abs(y - center_line)
        flood_mask = dist_to_river < (32 + 12 * np.sin(x / 20.0))
        
        # Elevated road corridor (N1 highway bridge section)
        bridge_mask = (np.abs(x - 128) < 8) & (dist_to_river < 44)
        flood_mask[bridge_mask] = False

        start_coord = (145, 30)
        goal_coord = (35, 220)

    elif scenario_type == "white_volta":
        scenario_name = "White Volta & Bagre Dam Overflow (Northern Region, Ghana)"
        desc = "Annual opening of the Bagre Dam combined with monsoon rain, flooding riverine farmland around the White Volta basin near Pwalugu."
        
        # Pwalugu / Walewale White Volta corridor
        bounds = [10.38, -0.92, 10.52, -0.78]
        center = [10.45, -0.85]
        start_label = "Pwalugu Lowland Farmland"
        goal_label = "Walewale District Hospital"

        # River floodplain
        river_line = 115 + 0.3 * x + 28 * np.sin(x / 35.0)
        flood_mask = (y > river_line - 18) & (y < river_line + 42)

        # Elevated N10 Tamale-Bolgatanga highway
        highway = (np.abs(y - 120) < 6) & (x > 50) & (x < 205)
        flood_mask[highway] = False

        start_coord = (180, 35)
        goal_coord = (40, 215)

    else:  # Greater Accra flash flood
        scenario_name = "Greater Accra Flash Flood (Odaw River Basin, Ghana)"
        desc = "Intense seasonal rainfall overflowing the Odaw drain and Korle Lagoon, flooding low-lying areas around Kwame Nkrumah Circle."
        
        # Kwame Nkrumah Circle / Odaw Basin / Alajo
        bounds = [5.54, -0.25, 5.62, -0.16]
        center = [5.58, -0.21]
        start_label = "Kwame Nkrumah Circle / Alajo"
        goal_label = "Ridge Hospital Emergency Complex"

        # Odaw drainage corridor
        drain_line = 130 + 45 * np.sin(x / 26.0)
        dist_drain = np.abs(y - drain_line)
        flood_mask = dist_drain < (26 + 10 * np.cos(x / 18.0))

        # Elevated Ring Road / Independence Avenue corridor
        ridge_road = np.abs(y - 70) < 6
        flood_mask[ridge_road] = False

        start_coord = (165, 45)
        goal_coord = (35, 130)

    # Multi-spectral satellite simulation
    base_r = np.random.normal(88, 7, (h, w))
    base_g = np.random.normal(132, 10, (h, w))
    base_b = np.random.normal(68, 8, (h, w))

    # Roads
    roads = (np.abs(y - 110) < 3) | (np.abs(x - 128) < 3)
    base_r[roads] = 165
    base_g[roads] = 165
    base_b[roads] = 165

    # Turbid water (sediment-rich floodwater)
    turbidity = np.random.normal(0, 5, (h, w))
    base_r[flood_mask] = 34 + turbidity[flood_mask]
    base_g[flood_mask] = 68 + turbidity[flood_mask]
    base_b[flood_mask] = 115 + turbidity[flood_mask]

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
        "filename": f"ghana_{scenario_type}_sentinel.tif",
        "dimensions": (h, w),
        "ground_truth_flood": flood_mask,
        "recommended_start": start_coord,
        "recommended_goal": goal_coord,
        "geo_bounds": bounds,
        "geo_center": center,
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
    """
    s_lat, w_lon, n_lat, e_lon = geo_bounds
    lat = n_lat - (row / height) * (n_lat - s_lat)
    lon = w_lon + (col / width) * (e_lon - w_lon)
    return round(float(lat), 6), round(float(lon), 6)
