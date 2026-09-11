"""
RS-Flood-Mapper: Emergency Satellite Flood Intelligence & Safe Evacuation Routing System.
A high-performance remote sensing analytics dashboard for disaster management & spatial data engineering.
Features: Multi-format Ingestion, ML Inundation Delineation, OpenStreetMap Integration, Light/Dark Modes, Mobile Responsive.
"""

import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Geospatial and interactive map modules
from streamlit_folium import st_folium

# Project modules
from utils.image_handler import load_image_any_format
from utils.safety_analyzer import (
    compute_safety_zones,
    find_safe_evacuation_path,
    create_safety_overlay,
    calculate_disaster_telemetry,
)
from utils.demo_samples import generate_demo_scenario, pixel_to_latlon
from utils.map_renderer import generate_interactive_osm_map
from utils.benchmarking import benchmark_all_models, extract_feature_importances
from models.random_forest import get_default_rf_model, predict_with_rf

# Optional Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Page Configuration
st.set_page_config(
    page_title="RS Flood Mapper • Disaster Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_styles(is_dark: bool):
    """
    Inject responsive CSS tailored for Light or Dark theme with mobile-optimized layout.
    """
    if is_dark:
        bg_primary = "#0b0f19"
        bg_card = "linear-gradient(135deg, #1e293b 0%, #0f172a 100%)"
        border_color = "#334155"
        text_primary = "#f3f4f6"
        text_secondary = "#94a3b8"
        box_shadow = "0 4px 14px rgba(0,0,0,0.3)"
        briefing_bg = "#0f172a"
    else:
        bg_primary = "#f8fafc"
        bg_card = "linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%)"
        border_color = "#cbd5e1"
        text_primary = "#0f172a"
        text_secondary = "#64748b"
        box_shadow = "0 2px 10px rgba(0,0,0,0.06)"
        briefing_bg = "#e2e8f0"

    css = f"""
    <style>
        /* Mobile-Responsive Base Layout */
        .main .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }}
        
        /* Mission-Control HUD Grid */
        .hud-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 18px;
        }}
        
        .hud-card {{
            background: {bg_card};
            border-radius: 12px;
            padding: 16px 14px;
            border: 1px solid {border_color};
            box-shadow: {box_shadow};
            text-align: center;
            transition: transform 0.2s ease;
        }}
        .hud-card:hover {{
            transform: translateY(-2px);
        }}
        .hud-title {{
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {text_secondary};
            margin-bottom: 4px;
        }}
        .hud-value {{
            font-size: 1.75rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.2;
        }}
        .hud-subtitle {{
            font-size: 0.76rem;
            color: {text_secondary};
            margin-top: 4px;
        }}
        
        .badge-critical {{ color: #ef4444 !important; }}
        .badge-elevated {{ color: #f97316 !important; }}
        .badge-moderate {{ color: #eab308 !important; }}
        .badge-safe {{ color: #22c55e !important; }}
        .badge-cyan {{ color: #0284c7 !important; }}
        
        .briefing-box {{
            background: {briefing_bg};
            border-left: 4px solid #0284c7;
            padding: 14px 18px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 15px;
            color: {text_primary};
        }}

        /* Mobile Viewport Optimizations */
        @media (max-width: 900px) {{
            .hud-grid {{
                grid-template-columns: repeat(2, 1fr) !important;
                gap: 10px !important;
            }}
            .hud-card {{
                padding: 12px 10px !important;
            }}
            .hud-value {{
                font-size: 1.4rem !important;
            }}
            .stTabs [data-baseweb="tab"] {{
                padding: 8px 10px !important;
                font-size: 0.82rem !important;
            }}
        }}

        @media (max-width: 480px) {{
            .hud-grid {{
                grid-template-columns: 1fr !important;
                gap: 8px !important;
            }}
            .main .block-container {{
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }}
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_resource
def load_rf_classifier():
    """Load or train the cached baseline Random Forest classifier."""
    return get_default_rf_model(num_features=4)


def main():
    # Sidebar
    with st.sidebar:
        st.header("Interface Theme")
        theme_choice = st.radio(
            "Theme",
            ["Dark Mode", "Light Mode"],
            index=0
        )
        is_dark = "Dark" in theme_choice
        theme_str = "Dark" if is_dark else "Light"

        st.markdown("---")
        st.header("Data Source")
        
        input_source = st.radio(
            "Source",
            ["Ghana Flood Scenarios", "Upload Satellite Imagery"],
            index=0
        )

        rgb_img = None
        features = None
        meta = {}
        ground_truth = None
        default_start = (140, 20)
        default_goal = (30, 220)

        if input_source == "Ghana Flood Scenarios":
            scenario = st.selectbox(
                "Flood Scenario",
                [
                    "🇬🇭 Akosombo Dam Spillage (Volta Region - 2023)",
                    "🇬🇭 White Volta & Bagre Dam Overflow (Northern Region)",
                    "🇬🇭 Greater Accra Flash Flood (Odaw Basin & Circle)"
                ],
                index=0
            )
            key_map = {
                "🇬🇭 Akosombo Dam Spillage (Volta Region - 2023)": "volta",
                "🇬🇭 White Volta & Bagre Dam Overflow (Northern Region)": "white_volta",
                "🇬🇭 Greater Accra Flash Flood (Odaw Basin & Circle)": "accra"
            }
            scenario_key = key_map[scenario]
            rgb_img, features, meta = generate_demo_scenario(scenario_key)
            ground_truth = meta.get("ground_truth_flood")
            default_start = meta.get("recommended_start", (140, 20))
            default_goal = meta.get("recommended_goal", (30, 220))
            
            st.info(f"**Context:** {meta.get('description', '')}")

        else:
            uploaded_file = st.file_uploader(
                "Upload Imagery",
                type=["tif", "tiff", "png", "jpg", "jpeg", "webp", "bmp", "npy", "npz"],
                help="Accepts GeoTIFF, PNG, JPG, WEBP, BMP, and NumPy arrays (single-band SAR, RGB optical, or 4-band multi-spectral)."
            )
            if uploaded_file is not None:
                try:
                    rgb_img, features, meta = load_image_any_format(uploaded_file)
                    st.success(f"Ingested `{uploaded_file.name}` ({meta['dimensions'][1]}x{meta['dimensions'][0]} px, {meta['original_channels']} band(s))")
                    default_start = (int(meta['dimensions'][0] * 0.8), int(meta['dimensions'][1] * 0.2))
                    default_goal = (int(meta['dimensions'][0] * 0.2), int(meta['dimensions'][1] * 0.8))
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                st.warning("Please upload a satellite or aerial image to begin.")

        st.markdown("---")
        st.header("🔬 Inundation Modeling")
        
        detection_method = st.selectbox(
            "Classification Model",
            ["Random Forest (Machine Learning)", "NDWI Spectral Cutoff", "Adaptive Otsu Thresholding"],
            index=0
        )

        water_threshold = st.slider(
            "Spectral Sensitivity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5 if detection_method != "NDWI Spectral Cutoff" else 0.15,
            step=0.02,
            help="Higher threshold requires stronger spectral evidence of water absorption."
        )

        st.markdown("---")
        st.header("🛡️ Safe Passage & Routing")
        
        buffer_radius = st.slider(
            "Safety Hazard Margin (pixels)",
            min_value=3,
            max_value=35,
            value=12,
            step=1,
            help="Perimeter distance surrounding flood boundaries where bank failure or wave surge presents danger."
        )

        enable_routing = st.checkbox("Enable Evacuation Route Planner", value=True)

        start_pt = default_start
        goal_pt = default_goal

        if enable_routing and rgb_img is not None:
            h, w = rgb_img.shape[:2]
            with st.expander("📍 Waypoint Coordinates", expanded=False):
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    start_x = st.number_input("Origin X", 0, w - 1, value=min(int(default_start[1]), w - 1))
                with col_s2:
                    start_y = st.number_input("Origin Y", 0, h - 1, value=min(int(default_start[0]), h - 1))
                
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    goal_x = st.number_input("Shelter X", 0, w - 1, value=min(int(default_goal[1]), w - 1))
                with col_g2:
                    goal_y = st.number_input("Shelter Y", 0, h - 1, value=min(int(default_goal[0]), h - 1))

                start_pt = (start_y, start_x)
                goal_pt = (goal_y, goal_x)

    # Apply Responsive Theme Styles
    inject_custom_styles(is_dark)

    # Top Header
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 12px;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2rem;">🌊</span>
            <div>
                <h1 style="margin: 0; font-size: 1.7rem; font-weight: 700;">
                    Remote Sensing Flood Mapper
                </h1>
                <p style="margin: 0; font-size: 0.88rem; opacity: 0.85;">
                    Satellite Inundation Detection & Safe Route Navigation
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if rgb_img is None or features is None:
        st.info("👋 Select a disaster scenario in the sidebar or upload satellite imagery to start analysis.")
        return

    # Execute Inundation Analysis
    with st.spinner("Analyzing spectral signatures and computing hazard perimeters..."):
        h, w = rgb_img.shape[:2]
        rf_model = load_rf_classifier()
        
        if detection_method == "Random Forest (Machine Learning)":
            pred_mask = predict_with_rf(rf_model, features, (h, w))
            flood_mask = (pred_mask > 0).astype(bool)
        elif detection_method == "NDWI Spectral Cutoff":
            ndwi = features[:, :, 3] if features.shape[2] >= 4 else features[:, :, 0]
            flood_mask = ndwi >= water_threshold
        else:  # Otsu Adaptive Thresholding
            ndwi = features[:, :, 3] if features.shape[2] >= 4 else features[:, :, 0]
            norm_ndwi = ((ndwi + 1.0) / 2.0 * 255.0).astype(np.uint8)
            hist, _ = np.histogram(norm_ndwi, bins=256, range=(0, 256))
            total = float(norm_ndwi.size)
            current_max, threshold = 0.0, 128
            sum_total = np.dot(np.arange(256), hist)
            weight_bg, sum_bg = 0.0, 0.0
            
            for t in range(256):
                weight_bg += hist[t]
                if weight_bg == 0:
                    continue
                weight_fg = total - weight_bg
                if weight_fg == 0:
                    break
                sum_bg += t * hist[t]
                mean_bg = sum_bg / weight_bg
                mean_fg = (sum_total - sum_bg) / weight_fg
                var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
                if var_between > current_max:
                    current_max = var_between
                    threshold = t
            otsu_cutoff = (threshold / 255.0) * 2.0 - 1.0
            flood_mask = ndwi >= (otsu_cutoff * water_threshold)

        # Spatial Safety & Zonation Analysis
        safety_data = compute_safety_zones(flood_mask, buffer_distance=buffer_radius)
        zone_map = safety_data["zone_map"]
        distance_map = safety_data["distance_map"]
        telemetry = calculate_disaster_telemetry(flood_mask, zone_map)

        # A* Evacuation Routing
        evac_path = None
        route_status = "N/A"
        route_length = 0
        if enable_routing:
            evac_path = find_safe_evacuation_path(
                flood_mask, distance_map, start_pt, goal_pt, buffer_distance=buffer_radius
            )
            if evac_path:
                route_status = "PASSABLE 🟢"
                route_length = len(evac_path)
            else:
                route_status = "BLOCKED 🔴"

    # Responsive Telemetry HUD
    badge_flood = "badge-critical" if telemetry["flood_pct"] > 30 else ("badge-elevated" if telemetry["flood_pct"] > 15 else "badge-safe")
    badge_risk = "badge-critical" if "CRITICAL" in telemetry["risk_level"] else ("badge-elevated" if "ELEVATED" in telemetry["risk_level"] else "badge-safe")
    badge_route = "badge-safe" if "PASSABLE" in route_status else ("badge-critical" if "BLOCKED" in route_status else "")

    st.markdown(f"""
    <div class="hud-grid">
        <div class="hud-card">
            <div class="hud-title">Inundated Area</div>
            <div class="hud-value {badge_flood}">{telemetry['flood_pct']}%</div>
            <div class="hud-subtitle">~{telemetry['flood_area_ha']} Hectares Submerged</div>
        </div>
        <div class="hud-card">
            <div class="hud-title">Safe Traversable Land</div>
            <div class="hud-value badge-safe">{telemetry['safe_pct']}%</div>
            <div class="hud-subtitle">~{telemetry['safe_area_ha']} Ha Dry Corridor</div>
        </div>
        <div class="hud-card">
            <div class="hud-title">Disaster Hazard Rating</div>
            <div class="hud-value {badge_risk}" style="font-size: 1.25rem; line-height: 2rem;">{telemetry['risk_level']}</div>
            <div class="hud-subtitle">Buffer Margin: {telemetry['caution_pct']}%</div>
        </div>
        <div class="hud-card">
            <div class="hud-title">Evacuation Status</div>
            <div class="hud-value {badge_route}" style="font-size: 1.2rem; line-height: 2rem;">{route_status}</div>
            <div class="hud-subtitle">Route Length: {route_length} px</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Tabs Layout
    tab_osm, tab_inspect, tab_safety, tab_benchmarks, tab_analytics, tab_export = st.tabs([
        "🗺️ OpenStreetMap & Live Route",
        "🛰️ Satellite & Flood Extent",
        "🛡️ Safety Zonation Map",
        "🧠 Model Benchmarks & XAI",
        "📊 Spatial Analytics",
        "💾 Export Situation Briefings"
    ])

    # Tab 0: Interactive OpenStreetMap
    with tab_osm:
        st.subheader("OpenStreetMap Evacuation Route")
        st.markdown(r"""
        Georeferenced satellite flood layer overlaid on OpenStreetMap. 
        The cyan path shows the calculated safe evacuation route connecting the origin to the shelter.
        """)

        col_m1, col_m2 = st.columns([4, 1])
        with col_m2:
            st.markdown("#### Route Details")
            geo_bounds = meta.get("geo_bounds", [5.92, 0.50, 6.08, 0.68])
            s_row, s_col = start_pt
            g_row, g_col = goal_pt
            s_lat, s_lon = pixel_to_latlon(s_row, s_col, h, w, geo_bounds)
            g_lat, g_lon = pixel_to_latlon(g_row, g_col, h, w, geo_bounds)

            st.write(f"**Origin (Distress):** `{s_lat:.4f}, {s_lon:.4f}`")
            st.write(f"**Relief Shelter:** `{g_lat:.4f}, {g_lon:.4f}`")
            st.write(f"**Corridor Status:** {route_status}")
            st.write(f"**Waypoints:** {route_length} coordinates")
            st.caption("Tip: Use the top-right layer switch on the map to toggle between OpenStreetMap Streets and High-Resolution Satellite imagery.")

        with col_m1:
            folium_map = generate_interactive_osm_map(
                flood_mask, zone_map, meta, evac_path=evac_path if enable_routing else None,
                theme_mode=theme_str, buffer_radius=buffer_radius
            )
            st_folium(folium_map, use_container_width=True, height=520)

    # Tab 1: Satellite & Flood Extent
    with tab_inspect:
        col_view1, col_view2 = st.columns([1, 1])
        with col_view1:
            st.subheader("Raw Remote Sensing Image (RGB Composite)")
            st.image(rgb_img, caption="Multi-band Satellite Ingestion", use_container_width=True)

        with col_view2:
            st.subheader("Classified Inundation Boundary")
            view_mode = st.radio(
                "Display Mode",
                ["Color Inundation Overlay", "Split Screen Comparison", "Binary Flood Mask"],
                horizontal=True
            )

            if view_mode == "Color Inundation Overlay":
                flood_overlay = rgb_img.copy().astype(np.float32)
                water_color = np.array([30, 144, 255], dtype=np.float32)
                flood_overlay[flood_mask] = 0.45 * flood_overlay[flood_mask] + 0.55 * water_color
                st.image(np.clip(flood_overlay, 0, 255).astype(np.uint8), caption="Cyan Highlight: Classified Inundation", use_container_width=True)

            elif view_mode == "Split Screen Comparison":
                split_img = rgb_img.copy()
                mid = w // 2
                water_tint = np.array([235, 60, 60], dtype=np.uint8)
                split_img[:, mid:][flood_mask[:, mid:]] = water_tint
                st.image(split_img, caption="Left: Satellite True Color | Right: Classified Water Boundary", use_container_width=True)

            else:
                mask_display = (flood_mask.astype(np.uint8) * 255)
                st.image(mask_display, caption="Binary Water Mask (White = Water, Black = Dry Land)", use_container_width=True)

    # Tab 2: Safety Zonation Map
    with tab_safety:
        st.subheader("🛡️ Hazard Zonation & Evacuation Corridor Optimization")
        st.markdown(r"""
        Terrain is dynamically classified using Euclidean distance transforms:
        - 🔴 **Inundated Zone (Hazardous)**: Impassable water basins.
        - 🟡 **Caution Margin**: Dry terrain within risk perimeter of floodwaters (susceptible to sudden bank failure).
        - 🟢 **Safe Passageway**: High-clearance dry ground verified for vehicular and pedestrian transit.
        - ⚡ **Cyan Path**: Shortest safe evacuation path computed via **A* Graph Search** with safety cost-surface penalization.
        """)

        safety_rgb = create_safety_overlay(
            rgb_img, zone_map, path=evac_path if enable_routing else None, alpha=0.48
        )

        col_safe1, col_safe2 = st.columns([3, 2])
        with col_safe1:
            st.image(safety_rgb, caption="Safe Passageway Map (Green: Safe | Yellow: Caution | Red: Inundated | Cyan: Evacuation Route)", use_container_width=True)

        with col_safe2:
            st.subheader("Euclidean Safety Distance Heatmap")
            fig_heat, ax_heat = plt.subplots(figsize=(5, 4.5), facecolor=("#0b0f19" if is_dark else "#f8fafc"))
            ax_heat.set_facecolor("#0b0f19" if is_dark else "#f8fafc")
            im = ax_heat.imshow(distance_map, cmap="viridis", interpolation="bilinear")
            cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
            cbar.set_label("Clearance to Water (pixels)", color=("#ffffff" if is_dark else "#000000"), size=9)
            cbar.ax.tick_params(colors=("#ffffff" if is_dark else "#000000"), labelsize=8)
            ax_heat.axis("off")
            ax_heat.set_title("Clearance From Flood Hazard", color=("#ffffff" if is_dark else "#000000"), fontsize=11)
            st.pyplot(fig_heat)
            plt.close(fig_heat)

    # Tab 3: Model Benchmarks & XAI
    with tab_benchmarks:
        st.subheader("🧠 Model Performance Benchmarking & Explainable AI (XAI)")
        benchmark_target = ground_truth if ground_truth is not None else flood_mask

        col_bm1, col_bm2 = st.columns([3, 2])
        with col_bm1:
            st.markdown("#### Quantitative Performance Matrix")
            df_bench = benchmark_all_models(features, benchmark_target, rf_model, water_threshold=water_threshold)
            st.dataframe(df_bench.style.highlight_max(axis=0, subset=["mIoU", "Dice / F1", "Accuracy"], color="#0369a1"), use_container_width=True)

            if PLOTLY_AVAILABLE:
                fig_bar = px.bar(
                    df_bench,
                    x="Model",
                    y=["mIoU", "Dice / F1", "Accuracy"],
                    barmode="group",
                    title="Model Overlap & Accuracy Comparison",
                    color_discrete_sequence=["#0284c7", "#16a34a", "#9333ea"]
                )
                fig_bar.update_layout(
                    paper_bgcolor=("rgba(0,0,0,0)" if is_dark else "#ffffff"),
                    plot_bgcolor=("rgba(0,0,0,0)" if is_dark else "#f8fafc"),
                    font_color=("#ffffff" if is_dark else "#0f172a"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with col_bm2:
            st.markdown("#### Explainable AI: Spectral Band Weights")
            df_fi = extract_feature_importances(rf_model)
            if not df_fi.empty:
                st.dataframe(df_fi, use_container_width=True)
                if PLOTLY_AVAILABLE:
                    fig_fi = px.pie(
                        df_fi,
                        names="Spectral Band",
                        values="Importance (%)",
                        hole=0.45,
                        color_discrete_sequence=["#0284c7", "#16a34a", "#d97706", "#db2777"]
                    )
                    fig_fi.update_layout(
                        paper_bgcolor=("rgba(0,0,0,0)" if is_dark else "#ffffff"),
                        font_color=("#ffffff" if is_dark else "#0f172a"),
                        title="Spectral Feature Importance"
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

    # Tab 4: Spatial Analytics
    with tab_analytics:
        st.subheader("📊 Quantitative Terrain Distribution & Spectral Histograms")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Land Cover Classification")
            if PLOTLY_AVAILABLE:
                labels = ["Safe Ground", "Caution Buffer", "Flooded Area"]
                values = [telemetry["safe_pct"], telemetry["caution_pct"], telemetry["flood_pct"]]
                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.5,
                    marker=dict(colors=["#16a34a", "#d97706", "#dc2626"])
                )])
                fig_donut.update_layout(
                    paper_bgcolor=("rgba(0,0,0,0)" if is_dark else "#ffffff"),
                    font_color=("#ffffff" if is_dark else "#0f172a"),
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        with col_c2:
            st.markdown("#### NDWI Index Distribution & Cutoff")
            ndwi_vals = features[:, :, 3].flatten() if features.shape[2] >= 4 else features[:, :, 0].flatten()
            if PLOTLY_AVAILABLE:
                fig_hist = px.histogram(
                    x=ndwi_vals,
                    nbins=50,
                    labels={"x": "Normalized Difference Water Index (NDWI)"},
                    color_discrete_sequence=["#0284c7"]
                )
                fig_hist.add_vline(x=water_threshold if detection_method == "NDWI Spectral Cutoff" else 0.2, line_dash="dash", line_color="#dc2626", annotation_text="Water Threshold")
                fig_hist.update_layout(
                    paper_bgcolor=("rgba(0,0,0,0)" if is_dark else "#ffffff"),
                    plot_bgcolor=("rgba(0,0,0,0)" if is_dark else "#f8fafc"),
                    font_color=("#ffffff" if is_dark else "#0f172a"),
                    margin=dict(t=20, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # Tab 5: Export Situation Briefings
    with tab_export:
        st.subheader("💾 Export Maps, Geospatial Masks & Situation Reports")
        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            st.markdown("##### 1. Binary Flood Mask")
            mask_img = Image.fromarray((flood_mask * 255).astype(np.uint8))
            buf_mask = io.BytesIO()
            mask_img.save(buf_mask, format="PNG")
            st.download_button(
                label="📥 Download Flood Mask (PNG)",
                data=buf_mask.getvalue(),
                file_name=f"flood_mask_{meta.get('filename', 'scene')}.png",
                mime="image/png",
                use_container_width=True
            )

        with col_d2:
            st.markdown("##### 2. Safe Passageway Map")
            safe_pil = Image.fromarray(safety_rgb)
            buf_safe = io.BytesIO()
            safe_pil.save(buf_safe, format="PNG")
            st.download_button(
                label="📥 Download Safety Map (PNG)",
                data=buf_safe.getvalue(),
                file_name=f"safety_zones_{meta.get('filename', 'scene')}.png",
                mime="image/png",
                use_container_width=True
            )

        with col_d3:
            st.markdown("##### 3. Situation Report (JSON)")
            incident_data = {
                "system": "RS Flood Mapper & Safe Route Navigator",
                "scene_filename": meta.get("filename", "unknown"),
                "classification_architecture": detection_method,
                "hazard_risk_level": telemetry["risk_level"],
                "inundated_area_pct": telemetry["flood_pct"],
                "inundated_area_hectares": telemetry["flood_area_ha"],
                "safe_passageway_pct": telemetry["safe_pct"],
                "caution_buffer_pct": telemetry["caution_pct"],
                "evacuation_corridor_status": route_status,
                "evacuation_route_waypoints": route_length,
                "situation_assessment": telemetry["status_msg"]
            }
            json_report = json.dumps(incident_data, indent=2)
            st.download_button(
                label="📥 Download Situation Report (JSON)",
                data=json_report,
                file_name="flood_incident_report.json",
                mime="application/json",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
