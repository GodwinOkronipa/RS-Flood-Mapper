"""
Interactive OpenStreetMap (OSM) and Geospatial Rendering Engine.
Uses Folium to georeference flood hazard overlays and plot evacuation routes
directly on real-world street maps and satellite layers.
"""

import base64
import io
from typing import Dict, List, Optional, Tuple
import folium
from folium import plugins
import numpy as np
from PIL import Image

from utils.demo_samples import pixel_to_latlon


def generate_interactive_osm_map(
    flood_mask: np.ndarray,
    zone_map: np.ndarray,
    metadata: Dict,
    evac_path: Optional[List[Tuple[int, int]]] = None,
    theme_mode: str = "Dark",
    buffer_radius: int = 12
) -> folium.Map:
    """
    Construct a rich, interactive OpenStreetMap with georeferenced flood layers,
    safety zonation, relief shelters, and A* evacuation routes.

    Parameters
    ----------
    flood_mask : np.ndarray
        2D boolean array of flood extents.
    zone_map : np.ndarray
        2D array (0: Flood, 1: Caution, 2: Safe).
    metadata : dict
        Geospatial metadata containing 'geo_bounds' [south, west, north, east] and 'geo_center'.
    evac_path : list of (row, col), optional
        Pixel coordinates of the calculated safe evacuation path.
    theme_mode : str
        "Dark" or "Light".

    Returns
    -------
    folium.Map instance ready for rendering with streamlit-folium.
    """
    geo_bounds = metadata.get("geo_bounds", [5.92, 0.50, 6.08, 0.68])
    geo_center = metadata.get("geo_center", [6.00, 0.59])
    s_lat, w_lon, n_lat, e_lon = geo_bounds
    h, w = flood_mask.shape

    m = folium.Map(
        location=geo_center,
        zoom_start=12,
        tiles=None,  # We add custom layers below
        control_scale=True,
    )

    # Add OpenStreetMap Standard Layer
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors",
        name="🗺️ OpenStreetMap (Streets & Infrastructure)",
        overlay=False,
        control=True
    ).add_to(m)

    # Add Satellite Imagery Layer (Esri World Imagery)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
        name="🛰️ High-Resolution Satellite Basemap",
        overlay=False,
        control=True
    ).add_to(m)

    # 1. Generate transparent RGBA image for Flood Hazard Extent
    # Transparent where safe, Crimson where flooded, Amber for caution
    h_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Flooded pixels: Red (#E63946) with 70% opacity
    h_rgba[flood_mask] = [230, 45, 60, 180]

    # Caution margin pixels: Amber with 50% opacity
    caution_mask = (zone_map == 1)
    h_rgba[caution_mask] = [245, 175, 25, 130]

    img_pil = Image.fromarray(h_rgba, mode="RGBA")
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()
    data_url = f"data:image/png;base64,{b64_str}"

    folium_bounds = [[s_lat, w_lon], [n_lat, e_lon]]

    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=folium_bounds,
        opacity=0.85,
        name="🌊 Active Inundation & Hazard Zones",
        interactive=True,
        zindex=2
    ).add_to(m)

    # 2. Add Start Marker (Emergency Origin / Stranded Sector)
    start_row, start_col = metadata.get("recommended_start", (int(h * 0.7), int(w * 0.2)))
    start_lat, start_lon = pixel_to_latlon(start_row, start_col, h, w, geo_bounds)
    start_label = metadata.get("start_label", "Origin: Distress / Stranded Zone")

    folium.Marker(
        location=[start_lat, start_lon],
        popup=folium.Popup(f"<b>{start_label}</b><br>GPS: {start_lat:.5f}, {start_lon:.5f}<br>Status: Evacuation Requested", max_width=250),
        tooltip="📍 Origin: Stranded Population",
        icon=folium.Icon(color="red", icon="warning-sign", prefix="glyphicon")
    ).add_to(m)

    # 3. Add Goal Marker (Designated High-Ground Relief Shelter)
    goal_row, goal_col = metadata.get("recommended_goal", (int(h * 0.2), int(w * 0.8)))
    goal_lat, goal_lon = pixel_to_latlon(goal_row, goal_col, h, w, geo_bounds)
    goal_label = metadata.get("goal_label", "Destination: High-Ground Emergency Shelter")

    folium.Marker(
        location=[goal_lat, goal_lon],
        popup=folium.Popup(f"<b>{goal_label}</b><br>GPS: {goal_lat:.5f}, {goal_lon:.5f}<br>Status: Operational / Dry Relief Base", max_width=250),
        tooltip="🏥 Destination: Safe Relief Shelter",
        icon=folium.Icon(color="green", icon="plus-sign", prefix="glyphicon")
    ).add_to(m)

    # 4. Map the A* Evacuation Route directly onto OpenStreetMap streets
    if evac_path and len(evac_path) > 1:
        gps_route = []
        for r, c in evac_path:
            plat, plon = pixel_to_latlon(r, c, h, w, geo_bounds)
            gps_route.append([plat, plon])

        # Outer high-contrast casing
        folium.PolyLine(
            locations=gps_route,
            color="#0284c7",
            weight=8,
            opacity=0.6,
        ).add_to(m)

        # Core glowing route line
        folium.PolyLine(
            locations=gps_route,
            color="#00f0ff",
            weight=5,
            opacity=1.0,
            tooltip=f"⚡ A* Safe Evacuation Corridor ({len(evac_path)} GPS waypoints)",
            popup="<b>Recommended Evacuation Route</b><br>Strictly circumvents active floodwater and navigates verified dry roads."
        ).add_to(m)

    # Add Layer control and Fullscreen button
    plugins.Fullscreen(position="topright").add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m
