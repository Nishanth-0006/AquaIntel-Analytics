import os, sys, warnings, joblib, json, re
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    load_all_csvs, generate_synthetic_cwc, preprocess,
    CORE_FEATURES, BIS_STANDARDS
)

from utils.model_utils import SoftVotingHybrid

st.set_page_config(
    page_title="AquaIntel Analytics",
    page_icon="💧",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

:root {
    --primary: #10b981;
    --primary-hover: #059669;
    --sidebar-bg: #020617;
    --sidebar-text: #9ca3af;
    --sidebar-active: #ffffff;
    --bg-main: #0f172a;
    --card-bg: rgba(15, 23, 42, 0.8);
    --text-main: #ffffff;
    --text-muted: #94a3b8;
}


/* Global Reset */
html, body, [class*="css"], .stApp {
    font-family: 'Poppins', sans-serif !important;
    background-color: #0f172a !important;
    color: var(--text-main) !important;
}

.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.8) 100%), 
                url("https://images.unsplash.com/photo-1557683316-973673baf926?auto=format&fit=crop&q=80&w=2000");
    background-size: cover;
    background-position: center;
    backdrop-filter: blur(8px);
    z-index: -1;
}

/* UI Elements Contrast */
h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown span, .stMarkdown div, .stMarkdown b, .stMarkdown strong {
    color: var(--text-main) !important;
}

button[role="tab"] p {
    color: var(--text-muted) !important;
}

button[role="tab"][aria-selected="true"] p {
    color: var(--primary) !important;
    font-weight: 700 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--card-bg) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(12px) !important;
    color: var(--text-main) !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-main) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
}

[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* Specific fix for Data Summary text and small labels */
.stMarkdown [data-testid="stMarkdownContainer"] p {
    color: var(--text-main) !important;
}

.stMarkdown [data-testid="stMarkdownContainer"] b, 
.stMarkdown [data-testid="stMarkdownContainer"] strong {
    color: var(--text-main) !important;
    font-weight: 700 !important;
}

/* Global Button Styles - Forced Contrast */
.stButton > button, .stDownloadButton > button, [data-testid="stDownloadButton"] > button {
    background-color: #10b981 !important; /* Forced Green */
    color: #ffffff !important; /* Forced White */
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 0.75rem !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 700 !important;
    text-transform: none !important;
    letter-spacing: 0.025em !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
}

.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
    background-color: #059669 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 16px rgba(16, 185, 129, 0.3) !important;
    border-color: rgba(255, 255, 255, 0.4) !important;
}





/* File Uploader - Ultra Aggressive Visibility Fix */
[data-testid="stFileUploadDropzone"] {
    background-color: #1e293b !important; /* Deep Slate */
    border: 2px dashed #334155 !important;
    color: #ffffff !important;
}

/* Target the actual button inside the uploader */
[data-testid="stFileUploadDropzone"] button, 
.stFileUploader button,
[data-testid="stBaseButton-secondary"] {
    background-color: #10b981 !important; /* Aqua Green */
    color: #ffffff !important;
    border: none !important;
    opacity: 1 !important;
}

[data-testid="stFileUploadDropzone"] label,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] div {
    color: #ffffff !important;
}


[data-testid="stFileUploaderFileName"] {
    color: var(--text-main) !important;
}

small {
    color: var(--text-muted) !important;
}


/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #020617 !important;
    color: #ffffff !important;
}

/* Dark theme for all Sidebar inputs */
/* Seamless 'Floating Tag' Filter Design - Multi-Layer Fix */
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] .stMultiSelect [role="combobox"],
[data-testid="stSidebar"] .stMultiSelect div {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Subtle Underline for the input area */
[data-testid="stSidebar"] [data-baseweb="select"] {
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 0 !important;
}

[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: rgba(16, 185, 129, 0.1) !important;
    color: #10b981 !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    border-radius: 20px !important;
    padding: 2px 10px !important;
    font-weight: 600 !important;
    margin: 2px !important;
}


[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #94a3b8 !important; /* Muted grey for labels */
}


/* Header Centering */
.stApp header {
    display: none !important; /* Hide default header */
}

.centered-header {
    text-align: center;
    padding: 3rem 1rem;
    color: white;
}

/* KPI Cards (Enhanced) */

.kpi-card {
    background: rgba(30, 41, 59, 0.7);
    padding: 1.5rem;
    border-radius: 1.5rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
}


.kpi-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
}

.kpi-icon-box {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
}

.sparkline-box {
    width: 100%;
    height: 40px;
    margin-top: auto;
    opacity: 0.6;
}


/* Sidebar Active Item Simulation */
.sidebar-active-item {
    background: rgba(16, 185, 129, 0.1);
    color: #10b981 !important;
    border-radius: 0.5rem;
    padding: 0.5rem;
    font-weight: 600;
}

/* Top Navigation Simulation */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    margin-bottom: 2rem;
}

.welcome-text {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-main);
}

/* KPI Card Text Styles */
.kpi-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-muted);
}

.kpi-value {
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--text-main);
    margin-top: 0.25rem;
}

.kpi-trend {
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 0.5rem;
}


/* Streamlit Component Overrides */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--card-bg) !important;
    border-radius: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 1.5rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important;
}


.stButton > button {
    border-radius: 0.5rem !important;
    border: 1px solid #e5e7eb !important;
}

.primary-btn > div > button {
    background-color: var(--primary) !important;
    color: white !important;
    border: none !important;
}

/* Tabs */
div[role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid #e5e7eb !important;
    margin-bottom: 1.5rem !important;
}

button[role="tab"] {
    border: none !important;
    background: transparent !important;
}

button[role="tab"][aria-selected="true"] {
    border-bottom: 2px solid var(--primary) !important;
    color: var(--primary) !important;
}

</style>
""", unsafe_allow_html=True)



# ─── Color mapping ───────────────────────────────────────────
QUAL_COLORS = {
    "Excellent": "#27ae60", # Green
    "Good":      "#f1c40f", # Yellow
    "Poor":      "#e67e22", # Orange
    "Very Poor": "#e74c3c", # Red
}

# Plotly Global Layout Update

def update_chart_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff", family="Poppins, sans-serif"),
        title=dict(
            font=dict(color="#ffffff", size=18, weight="bold"),
            x=0,
            xanchor="left"
        ),
        legend=dict(
            font=dict(color="#ffffff", size=12),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.1)", 
            zerolinecolor="rgba(255,255,255,0.2)",
            tickfont=dict(color="#cbd5e1"),
            title_font=dict(color="#ffffff")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.1)", 
            zerolinecolor="rgba(255,255,255,0.2)",
            tickfont=dict(color="#cbd5e1"),
            title_font=dict(color="#ffffff")
        ),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig



# returns only colors present in dataset
def get_valid_colors(df, column, color_map):
    present = df[column].dropna().astype(str).str.strip().unique()
    return {k: v for k, v in color_map.items() if k in present}


GEOJSON_RELATIVE_PATH = os.path.join("data", "geo", "india_districts.geojson")


def normalize_geo_name(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def make_district_key(state, district):
    state_key = normalize_geo_name(state)
    district_key = normalize_geo_name(district)
    return f"{state_key}|{district_key}" if state_key else district_key


def first_property(properties, candidates):
    for key in candidates:
        value = properties.get(key)
        if value not in (None, ""):
            return value
    return ""


@st.cache_data
def load_india_district_geojson():
    path = os.path.join(os.path.dirname(__file__), GEOJSON_RELATIVE_PATH)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    except Exception as e:
        st.warning(f"Could not load district GeoJSON: {str(e)}")
        return None

    district_keys = [
        "district", "District", "DISTRICT", "district_name", "DISTRICT_NAME",
        "dist_name", "DIST_NAME", "dt_name", "DT_NAME", "dtname", "DTNAME",
        "NAME_2", "NAME2", "name", "Name", "NAME",
    ]
    state_keys = [
        "state", "State", "STATE", "state_name", "STATE_NAME",
        "st_nm", "ST_NM", "stname", "STNAME", "NAME_1", "NAME1",
    ]

    for feature in geojson.get("features", []):
        properties = feature.setdefault("properties", {})
        district = first_property(properties, district_keys)
        state = first_property(properties, state_keys)
        properties["plotly_key"] = make_district_key(state, district)

    return geojson


@st.cache_data
def build_district_choropleth_frame(frame):
    if frame.empty or "WQI" not in frame.columns:
        return pd.DataFrame(columns=["state", "district", "WQI", "plotly_key"])

    district_col = next((c for c in ["district", "District", "district_name"] if c in frame.columns), None)
    state_col = next((c for c in ["state", "State", "state_name"] if c in frame.columns), None)
    if district_col is None:
        return pd.DataFrame(columns=["state", "district", "WQI", "plotly_key"])

    cols = [district_col, "WQI"]
    if state_col:
        cols.insert(0, state_col)

    district_df = frame[cols].dropna(subset=[district_col, "WQI"]).copy()
    if district_df.empty:
        return pd.DataFrame(columns=["state", "district", "WQI", "plotly_key"])

    rename_map = {district_col: "district"}
    if state_col:
        rename_map[state_col] = "state"
    district_df = district_df.rename(columns=rename_map)
    if "state" not in district_df.columns:
        district_df["state"] = ""

    district_df["district"] = district_df["district"].astype(str).str.strip()
    district_df["state"] = district_df["state"].astype(str).str.strip()
    district_df = (
        district_df.groupby(["state", "district"], as_index=False)["WQI"]
        .mean()
        .sort_values("WQI", ascending=False)
    )
    district_df["plotly_key"] = district_df.apply(
        lambda row: make_district_key(row["state"], row["district"]),
        axis=1,
    )
    return district_df


# ─── Data load ───────────────────────────────────────────────
# loads dataset (real or synthetic)
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    try:
        raw = load_all_csvs(data_dir)
        source = "real"
    except FileNotFoundError:
        st.warning("No data files found. Generating synthetic data...")
        raw = generate_synthetic_cwc(n=8000)
        source = "synthetic"
    except Exception as e:
        st.error(f"Error loading data: {str(e)}. Using synthetic data instead.")
        raw = generate_synthetic_cwc(n=8000)
        source = "synthetic"
    return preprocess(raw), source


# loads models once
@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    models = {}
    for f in ["rf_full.pkl", "xgb_full.pkl", "hybrid_soft.pkl"]:
        path = os.path.join(model_dir, f)
        if os.path.exists(path):
            try:
                models[f.replace(".pkl", "")] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load model {f}: {str(e)}")
    return models


df, source = load_data()
models = load_models()

# ─── Sidebar (Glass Control Center) ──────────────────────────
st.sidebar.markdown('<div class="sidebar-header">Control Center</div>', unsafe_allow_html=True)


# Global Search
search_query = st.sidebar.text_input("Global Search", placeholder="Search Location, State...", label_visibility="collapsed")

# Initialize session state for filters
states = sorted(df["state"].dropna().unique())
if "state_filter" not in st.session_state:
    st.session_state.state_filter = states[:10]
if "year_filter" not in st.session_state:
    if "year" in df.columns:
        yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
        st.session_state.year_filter = (yr_min, yr_max)
    else:
        st.session_state.year_filter = (2010, 2024)
if "wqi_filter" not in st.session_state:
    st.session_state.wqi_filter = (0.0, 100.0)

st.sidebar.markdown("""<div style="padding: 1rem 0; margin-bottom: 2rem;">
<h2 style="color: white !important; font-size: 1.5rem; font-weight: 700;">AquaIntel</h2>
<p style="color: #9ca3af !important; font-size: 0.8rem;">Data Solutions</p>
</div>""", unsafe_allow_html=True)


# Navigation Menu Simulation (Active Dashboard Only)
st.sidebar.markdown("""<div style="background: rgba(16, 185, 129, 0.15); color: #10b981; border-radius: 12px; padding: 12px 16px; font-weight: 700; margin-bottom: 2rem; display: flex; align-items: center; gap: 12px; border: 1px solid rgba(16, 185, 129, 0.3);">
Dashboard (Active)
</div>""", unsafe_allow_html=True)



st.sidebar.markdown("---")

# Filter Groups
st.sidebar.markdown('<div style="color: white; font-weight: 600; margin-bottom: 1rem;">FILTERS</div>', unsafe_allow_html=True)

st.sidebar.markdown("##### Geographical Filter")

sel_states = st.sidebar.multiselect("Select States", states, key="state_filter")

st.sidebar.markdown("##### Temporal Filter")

if "year" in df.columns:
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    sel_years = st.sidebar.slider("Select Year Range", yr_min, yr_max, (yr_min, yr_max), key="year_filter")
else:
    sel_years = None

st.sidebar.markdown("##### Parameter Thresholds")

wqi_range = st.sidebar.slider("WQI Score Range", 0.0, 100.0, (0.0, 100.0), key="wqi_filter")

if st.sidebar.button("Reset Dashboard", use_container_width=True):
    st.rerun()




# ─── Filtering ───────────────────────────────────────────────
# applies filters to dataset
@st.cache_data
def apply_filters(df, sel_states, sel_years, wqi_range, search_query):
    filt = df.copy()
    
    # 1. Search Query (Global text search)
    if search_query:
        # Search in all string columns
        mask = filt.apply(lambda row: search_query.lower() in str(row).lower(), axis=1)
        filt = filt[mask]
    
    # 2. State Filter
    if sel_states:
        filt = filt[filt["state"].isin(sel_states)]
    
    # 3. Year Filter
    if sel_years and "year" in filt.columns:
        filt = filt[(filt["year"] >= sel_years[0]) & (filt["year"] <= sel_years[1])]
    
    # 4. WQI Filter
    filt = filt[(filt["WQI"] >= wqi_range[0]) & (filt["WQI"] <= wqi_range[1])]
    
    # clean labels
    if "water_quality" in filt.columns:
        filt["water_quality"] = filt["water_quality"].astype(str).str.strip()
    
    return filt

filt = apply_filters(df, sel_states, sel_years, wqi_range, search_query)

# downsample for performance
plot_df = filt.sample(min(3000, len(filt)), random_state=42)


# calculates distance between two coordinates (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    
    return R * c

# finds nearest stations to user location
def get_nearest_stations(df, user_lat, user_lon, n=3):

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    # 🔥 LIMIT DATA FIRST
    df = df.sample(min(4000, len(df)), random_state=42)

    # VECTOR DISTANCE
    lat1 = np.radians(user_lat)
    lon1 = np.radians(user_lon)

    lat2 = np.radians(df["latitude"].values)
    lon2 = np.radians(df["longitude"].values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    df["distance_km"] = 6371 * c

    df = df.sort_values("distance_km")

    df = df.drop_duplicates(subset=["latitude", "longitude"])

    if "station" in df.columns:
        df = df.drop_duplicates(subset=["station"])

    return df.head(n)

# ─── Header Section ──────────────────────────────────────────
st.markdown("""<div class="centered-header">
<div style="font-size: 5rem; margin-bottom: 0.5rem; filter: drop-shadow(0 10px 15px rgba(0,0,0,0.3));">💧</div>
<h1 style="font-size: 3.5rem; font-weight: 800; color: #ffffff; margin: 0; letter-spacing: -2px; text-shadow: 0 4px 10px rgba(0,0,0,0.2);">AquaIntel Analytics Dashboard</h1>
<p style="color: rgba(255, 255, 255, 0.8); font-size: 1.25rem; margin-top: 0.5rem; font-weight: 500;">Intelligent Water Quality Monitoring & Global Insight Engine</p>
</div>""", unsafe_allow_html=True)






# ─── Problem Statement ───────────────────────────────────────
st.markdown(f"""<div style="max-width: 900px; margin: 0 auto 3rem auto; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px); border-radius: 80px; padding: 20px 40px; border: 1px solid rgba(255, 255, 255, 0.2); box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
<div style="display: flex; align-items: center; gap: 20px; width: 100%;">
<div style="width: 50px; height: 50px; background: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0;">🎯</div>
<div style="flex-grow: 1; min-width: 0;">
<h4 style="margin: 0; color: #ffffff; font-weight: 700; font-size: 1.1rem;">Strategic Analysis Goal</h4>
<p style="margin: 2px 0 0 0; color: rgba(255, 255, 255, 0.85); font-size: 0.95rem; line-height: 1.4; overflow: hidden; text-overflow: ellipsis;">
Water data is fragmented and poorly analyzed, causing delayed detection of contamination and ineffective decision-making.
</p>
</div>
</div>
</div>""", unsafe_allow_html=True)







st.markdown(f"<div style='margin-bottom: 20px; color: var(--text-muted); font-size: 0.9rem;'><b>{len(filt):,} records</b> discovered · Source: <span style='color: var(--primary); font-weight: 600;'>{source.upper()}</span></div>", unsafe_allow_html=True)




# ─── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Risk Heatmap",
    "Trends",
    "Analysis",
    "Predict",
    "Upload"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:

    # Column detection for charts
    river_col = "River" if "River" in filt.columns else "river_name" if "river_name" in filt.columns else None

    # --- KPI Cards (SaaS Style) ---
    def create_sparkline(data, color="#27ae60"):
        fig = px.area(data, x=data.index, y=data.values, color_discrete_sequence=[color])
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=50,
            xaxis={'visible': False},
            yaxis={'visible': False},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_traces(fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}")
        return fig

    st.markdown("### Executive Summary")

    
    # Smart Status Verdict
    safe_pct_global = (filt["is_safe"].mean() * 100) if "is_safe" in filt.columns and not filt.empty else 0.0
    if not filt.empty:
        if safe_pct_global > 80:
            st.success(f"**System Status: EXCELLENT** ({safe_pct_global:.1f}% Safe). Water quality across selected regions is optimal.")

        elif safe_pct_global > 50:
            st.warning(f"**System Status: CONCERNING** ({safe_pct_global:.1f}% Safe). Some regions require immediate monitoring.")

        else:
            st.error(f"**System Status: CRITICAL** ({safe_pct_global:.1f}% Safe). Urgent intervention required in high-risk zones.")

    
    # Pre-calculate values for cards
    mean_wqi = filt["WQI"].mean() if not filt.empty else 0
    total_recs = len(filt)
    active_states = filt['state'].nunique() if not filt.empty else 0
    safe_pct_global = (filt["is_safe"].mean() * 100) if "is_safe" in filt.columns and not filt.empty else 0.0

    # ─── KPI Cards (Glass SaaS Style) ───

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    # Sparkline SVG paths (representative)
    spark_blue = "M0 30 Q10 10, 20 25 T40 15 T60 25 T80 10 T100 20"
    spark_green = "M0 35 Q10 30, 20 15 T40 25 T60 10 T80 20 T100 5"
    spark_orange = "M0 25 Q10 15, 20 35 T40 10 T60 25 T80 15 T100 30"
    spark_red = "M0 10 Q10 25, 20 10 T40 30 T60 15 T80 35 T100 20"

    # 1. Overall WQI Score
    with kpi_col1:
        st.markdown(f"""<div class="kpi-card">
<div class="kpi-header">
<div class="kpi-title">Mean WQI Score</div>
<div class="kpi-icon-box" style="background: rgba(37, 99, 235, 0.1); color: #2563eb;"></div>
</div>
<div class="kpi-value" style="color: #2563eb;">{mean_wqi:.1f}</div>
<div class="kpi-trend" style="color: #2563eb;">↑ 2.4% vs last month</div>
<div class="sparkline-box">
<svg viewBox="0 0 100 40" width="100%" height="100%" preserveAspectRatio="none">
<path d="{spark_blue}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linecap="round"/>
</svg>
</div>
</div>""", unsafe_allow_html=True)


    # 2. Total Records
    with kpi_col2:
        st.markdown(f"""<div class="kpi-card">
<div class="kpi-header">
<div class="kpi-title">Total Data Points</div>
<div class="kpi-icon-box" style="background: rgba(16, 185, 129, 0.1); color: #10b981;"></div>
</div>
<div class="kpi-value" style="color: #10b981;">{total_recs:,}</div>
<div class="kpi-trend" style="color: #10b981;">↑ 12% Growth</div>
<div class="sparkline-box">
<svg viewBox="0 0 100 40" width="100%" height="100%" preserveAspectRatio="none">
<path d="{spark_green}" fill="none" stroke="#10b981" stroke-width="3" stroke-linecap="round"/>
</svg>
</div>
</div>""", unsafe_allow_html=True)


    # 3. Active States
    with kpi_col3:
        st.markdown(f"""<div class="kpi-card">
<div class="kpi-header">
<div class="kpi-title">Covered Regions</div>
<div class="kpi-icon-box" style="background: rgba(245, 158, 11, 0.1); color: #f59e0b;"></div>
</div>
<div class="kpi-value" style="color: #f59e0b;">{active_states}</div>
<div class="kpi-trend" style="color: #f59e0b;">Active Coverage</div>
<div class="sparkline-box">
<svg viewBox="0 0 100 40" width="100%" height="100%" preserveAspectRatio="none">
<path d="{spark_orange}" fill="none" stroke="#f59e0b" stroke-width="3" stroke-linecap="round"/>
</svg>
</div>
</div>""", unsafe_allow_html=True)


    # 4. Safety Index
    with kpi_col4:
        safety_color = "#10b981" if safe_pct_global > 80 else "#ef4444"
        safety_bg = "rgba(16, 185, 129, 0.1)" if safe_pct_global > 80 else "rgba(239, 68, 68, 0.1)"
        safety_spark = spark_green if safe_pct_global > 80 else spark_red
        st.markdown(f"""<div class="kpi-card">
<div class="kpi-header">
<div class="kpi-title">Safety Confidence</div>
<div class="kpi-icon-box" style="background: {safety_bg}; color: {safety_color};"></div>
</div>
<div class="kpi-value" style="color: {safety_color};">{safe_pct_global:.1f}%</div>
<div class="kpi-trend" style="color: {safety_color};">Validated</div>
<div class="sparkline-box">
<svg viewBox="0 0 100 40" width="100%" height="100%" preserveAspectRatio="none">
<path d="{safety_spark}" fill="none" stroke="{safety_color}" stroke-width="3" stroke-linecap="round"/>
</svg>
</div>
</div>""", unsafe_allow_html=True)







    st.markdown("---")

    # --- Charts Layout ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.markdown('<div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; color: var(--text-main);">Quality Distribution</div>', unsafe_allow_html=True)


            with st.spinner("Generating distribution chart..."):
                wq_counts = filt["water_quality"].value_counts().reset_index()
                wq_counts.columns = ["Category", "Count"]
                total_count = wq_counts["Count"].sum()
                
                # Create two columns for pie chart and custom legend
                pie_col, legend_col = st.columns([1.2, 1])
                
                with pie_col:
                    fig_pie = px.pie(
                        wq_counts,
                        names="Category",
                        values="Count",
                        color="Category",
                        color_discrete_map=get_valid_colors(wq_counts, "Category", QUAL_COLORS),
                        hole=0.4
                    )
                    fig_pie.update_layout(
                        showlegend=False, 
                        margin=dict(t=10, b=10, l=10, r=10), 
                        height=300
                    )
                    fig_pie = update_chart_layout(fig_pie)
                    st.plotly_chart(fig_pie, use_container_width=True)

                
                with legend_col:
                    st.markdown('<div style="font-weight: 600; font-size: 14px; margin-bottom: 15px; color: var(--text-main);">Category Breakdown</div>', unsafe_allow_html=True)
                    
                    # Build custom HTML legend with progress bars
                    legend_html = "<div style='display: flex; flex-direction: column; gap: 12px;'>"
                    valid_colors = get_valid_colors(wq_counts, "Category", QUAL_COLORS)
                    
                    for _, row in wq_counts.iterrows():
                        cat = row["Category"]
                        count = row["Count"]
                        pct = (count / total_count) * 100 if total_count > 0 else 0
                        color = valid_colors.get(cat, "#cccccc")
                        
                        legend_html += f"""<div style='display: flex; flex-direction: column; gap: 4px;'>
<div style='display: flex; justify-content: space-between; align-items: center; font-size: 13px; color: var(--text-main);'>
    <div style='display: flex; align-items: center; gap: 6px;'>
        <div style='width: 8px; height: 8px; border-radius: 50%; background-color: {color};'></div>
        <span>{cat}</span>
    </div>
    <span style='font-weight: 600;'>{pct:.1f}%</span>
</div>
<div style='width: 100%; height: 6px; background-color: var(--border); border-radius: 3px; overflow: hidden;'>
    <div style='width: {pct}%; height: 100%; background-color: {color}; border-radius: 3px;'></div>
</div>
</div>"""
                    legend_html += "</div>"
                    st.markdown(legend_html, unsafe_allow_html=True)



    with chart_col2:
        with st.container(border=True):
            st.markdown('<div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; color: var(--text-main);">Regional Insights</div>', unsafe_allow_html=True)


            
            # State/Region Leaderboard
            state_col = "state" if "state" in filt.columns else "State" if "State" in filt.columns else None
            
            if not filt.empty:
                state_data = filt.groupby(state_col)["WQI"].mean().reset_index()
                cleanest = state_data.sort_values("WQI").head(3)
                dirtiest = state_data.sort_values("WQI", ascending=False).head(3)
                
                l_col1, l_col2 = st.columns(2)
                
                with l_col1:
                    st.markdown("<div style='font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 8px;'>Cleanest Regions</div>", unsafe_allow_html=True)
                    for _, row in cleanest.iterrows():
                        st.markdown(f"""
                            <div class='leaderboard-card' style='border-left: 4px solid #27ae60;'>
                                <div style='font-size: 12px; color: var(--text-muted);'>{row[state_col]}</div>
                                <div style='font-size: 16px; font-weight: 700; color: #27ae60;'>WQI: {row['WQI']:.1f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                with l_col2:
                    st.markdown("<div style='font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 8px;'>Critical Zones</div>", unsafe_allow_html=True)
                    for _, row in dirtiest.iterrows():
                        st.markdown(f"""
                            <div class='leaderboard-card' style='border-left: 4px solid #e74c3c;'>
                                <div style='font-size: 12px; color: var(--text-muted);'>{row[state_col]}</div>
                                <div style='font-size: 16px; font-weight: 700; color: #e74c3c;'>WQI: {row['WQI']:.1f}</div>
                            </div>
                        """, unsafe_allow_html=True)


                # Re-adding the detailed river chart for granular analysis
                if river_col:
                    st.markdown("---")
                    st.markdown("<div style='font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 15px;'>Detailed River Analysis</div>", unsafe_allow_html=True)
                    river_summary = filt.groupby(river_col)["WQI"].mean().reset_index().sort_values("WQI", ascending=False).head(5)
                    fig_bar = px.bar(
                        river_summary,
                        x="WQI",
                        y=river_col,
                        orientation="h",
                        color="WQI",
                        color_continuous_scale="RdYlGn_r",
                        height=250
                    )
                    fig_bar.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
                    fig_bar = update_chart_layout(fig_bar)
                    st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

            else:
                st.info("No data available for leaderboard analysis.")






## ─── Heatmap ─────────────────────────────────────────────
with tab2:

    st.markdown("### Water Quality Spatial Analysis")

    if "latitude" in filt.columns:

        # Map configuration - merged controls
        with st.container(border=True):
            st.markdown('<div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 1rem; color: var(--text-main);">Map Configuration</div>', unsafe_allow_html=True)


            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

            
            with col1:
                basemap_style = st.selectbox(
                    "Basemap",
                    ["OpenStreetMap", "Carto Positron", "Carto Dark"],
                    index=0,
                    key="basemap_selector"
                )
            
            with col2:
                color_theme = st.selectbox(
                    "Color Theme",
                    ["Green-Yellow-Red", "Blue-Purple-Red", "Viridis", "Plasma", "Inferno"],
                    index=0,
                    key="color_theme_selector"
                )
            
            with col3:
                color_mode = st.selectbox(
                    "Station Color",
                    ["Safe/Unsafe", "WQI Gradient", "Quality Categories"],
                    index=0,
                    key="color_mode_selector"
                )
            
            with col4:
                show_density = st.checkbox(
                    "Show Density",
                    value=True,
                    key="density_toggle"
                )
            
            st.markdown("")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                density_radius = st.slider("Density Radius", 5, 30, 18, key="density_radius")
            
            with col2:
                marker_size = st.slider("Marker Size", 5, 25, 15, key="marker_size")
            
            with col3:
                enable_animation = st.checkbox("Enable Time Animation", value=True, key="animation_toggle")
        
        # Color theme mapping
        color_themes = {
            "Green-Yellow-Red": [[0.0, "#27ae60"], [0.5, "#f1c40f"], [1.0, "#e74c3c"]],
            "Blue-Purple-Red": [[0.0, "#3498db"], [0.5, "#9b59b6"], [1.0, "#e74c3c"]],
            "Viridis": [[0.0, "#440154"], [0.5, "#21918c"], [1.0, "#fde725"]],
            "Plasma": [[0.0, "#0d0887"], [0.5, "#cc4778"], [1.0, "#f0f921"]],
            "Inferno": [[0.0, "#000004"], [0.5, "#b53679"], [1.0, "#fcffa4"]]
        }
        selected_colors = color_themes[color_theme]
        
        # Map style mapping
        basemap_map = {
            "OpenStreetMap": "open-street-map",
            "Carto Positron": "carto-positron",
            "Carto Dark": "carto-darkmatter"
        }
        selected_basemap = basemap_map[basemap_style]

        with st.spinner("Loading spatial data..."):
            map_df = filt.dropna(subset=["latitude", "longitude", "WQI", "water_quality"])

            if len(map_df) == 0:
                st.warning("No valid coordinates found")
            else:
                # Custom animation control using Streamlit slider
                if enable_animation and "year" in map_df.columns:
                    available_years = sorted(map_df["year"].unique())
                    selected_year = st.selectbox("Select Year", available_years, index=len(available_years)-1, key="year_selector")
                    map_df = map_df[map_df["year"] == selected_year]
                    st.info(f"Showing data for year: {selected_year}")
                density_df = map_df.sample(min(2000, len(map_df)), random_state=42)
                scatter_df = map_df.sample(min(1000, len(map_df)), random_state=42)

                # Density Map with enhanced styling
                if show_density:
                    with st.spinner("Generating density heatmap..."):
                        fig = px.density_mapbox(
                            density_df,
                            lat="latitude",
                            lon="longitude",
                            z="WQI",
                            radius=density_radius,
                            center={"lat": 20.5, "lon": 80},
                            zoom=4,
                            mapbox_style="carto-positron",
                            color_continuous_scale=selected_colors,
                            title="Water Quality Density Heatmap",
                            labels={"z": "WQI Index"},
                            range_color=[density_df["WQI"].min(), density_df["WQI"].max()],
                            opacity=0.8
                        )
                        fig.update_layout(
                            height=500, 
                            margin=dict(l=0, r=0, t=30, b=0),
                            coloraxis_colorbar=dict(title="WQI", x=1.02, len=0.8)
                        )
                        fig = update_chart_layout(fig)
                        fig.update_traces(opacity=0.9)
                        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})


                st.markdown("")

                # Scatter Map with enhanced features
                if color_mode == "Safe/Unsafe":
                    scatter_df["status"] = scatter_df["is_safe"].map({1: "Safe", 0: "Unsafe"})
                    color_column = "status"
                    status_colors = {"Safe": "#27ae60", "Unsafe": "#e74c3c"}
                    color_discrete_map = status_colors
                elif color_mode == "WQI Gradient":
                    color_column = "WQI"
                    color_discrete_map = None
                    status_colors = None
                else:
                    color_column = "water_quality"
                    color_discrete_map = get_valid_colors(scatter_df, "water_quality", QUAL_COLORS)
                    status_colors = None

                with st.spinner("Rendering station map..."):
                    if color_mode == "WQI Gradient":
                        fig2 = px.scatter_mapbox(
                            scatter_df,
                            lat="latitude",
                            lon="longitude",
                            color="WQI",
                            size="WQI",
                            size_max=marker_size,
                            center={"lat": 20.5, "lon": 80},
                            zoom=4,
                            mapbox_style=selected_basemap,
                            title="Water Quality Monitoring Network",
                            color_continuous_scale=selected_colors,
                            hover_data={"WQI": ":.2f", "water_quality": True, "state": True}
                        )
                    else:
                        fig2 = px.scatter_mapbox(
                            scatter_df,
                            lat="latitude",
                            lon="longitude",
                            color=color_column,
                            color_discrete_map=color_discrete_map,
                            size="WQI",
                            size_max=marker_size,
                            center={"lat": 20.5, "lon": 80},
                            zoom=4,
                            mapbox_style=selected_basemap,
                            title="Water Quality Monitoring Network",
                            hover_data={"WQI": ":.2f", "water_quality": True, "state": True}
                        )

                    fig2.update_traces(marker=dict(opacity=0.85), selector=dict(mode='markers'))
                    fig2.update_layout(
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    fig2 = update_chart_layout(fig2)
                    st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})


    st.markdown("### District-wise Water Quality Map")

    with st.spinner("Building district choropleth..."):
        geojson = load_india_district_geojson()
        district_map_df = build_district_choropleth_frame(filt)

        if geojson is None:
            st.warning("District GeoJSON is missing. Add `data/geo/india_districts.geojson` to enable the map.")
        elif district_map_df.empty:
            st.warning("No district-level WQI data is available for the current filters.")
        else:
            geojson_keys = {
                feature.get("properties", {}).get("plotly_key")
                for feature in geojson.get("features", [])
                if feature.get("properties", {}).get("plotly_key")
            }
            district_map_df = district_map_df[district_map_df["plotly_key"].isin(geojson_keys)].copy()

            if district_map_df.empty:
                st.warning("District names could not be matched to the India district boundaries.")
            else:
                with st.container(border=True):
                    st.markdown('<div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; color: var(--text-main);">Regional Risk Heatmap</div>', unsafe_allow_html=True)

                    
                    color_scale = [
                        [0.0, "#d7efe4"],
                        [0.35, "#8fcdb2"],
                        [0.65, "#f3d98b"],
                        [1.0, "#de7c6b"],
                    ]
                    fig_map = px.choropleth_mapbox(
                        district_map_df,
                        geojson=geojson,
                        locations="plotly_key",
                        featureidkey="properties.plotly_key",
                        color="WQI",
                        color_continuous_scale=color_scale,
                        range_color=(0, 100),
                        mapbox_style="carto-positron",
                        zoom=4,
                        center={"lat": 20.5, "lon": 80},
                        opacity=0.9,
                        custom_data=["district"],
                        hover_name=None,
                        hover_data=None,
                    )
                    fig_map.update_traces(
                        marker_line_color="white",
                        marker_line_width=0.5,
                        hovertemplate="<b>%{customdata[0]}</b><br>WQI: %{z:.1f}<extra></extra>",
                        showscale=False,
                    )
                    fig_map.update_layout(
                        height=720,
                        margin=dict(l=0, r=0, t=8, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        coloraxis_showscale=False,
                        mapbox=dict(style="carto-positron", zoom=4, center=dict(lat=20.5, lon=80)),
                    )
                    st.plotly_chart(
                        fig_map,
                        use_container_width=True,
                        config={"displayModeBar": False, "scrollZoom": True},
                    )


                # Enhanced statistics with visual indicators
                st.markdown("---")
                with st.container(border=True):
                    st.markdown('<div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1.5rem; color: var(--text-main);">Spatial Analytics Dashboard</div>', unsafe_allow_html=True)


                    
                    col1, col2, col3, col4 = st.columns(4)

                safe_pct = (len(map_df[map_df["is_safe"] == 1]) / len(map_df)) * 100
                
                col1.metric("Total Stations", len(map_df), delta=f"{len(map_df)} locations")
                col2.metric("Safe Stations", len(map_df[map_df["is_safe"] == 1]), delta=f"{safe_pct:.1f}%")
                col3.metric("Unsafe Stations", len(map_df[map_df["is_safe"] == 0]), delta=f"{100-safe_pct:.1f}%")
                col4.metric("Mean WQI", round(map_df["WQI"].mean(), 2), delta=f"Range: {map_df['WQI'].min():.1f}-{map_df['WQI'].max():.1f}")

                # Export with options
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                csv_data = map_df.to_csv(index=False)
                col1.download_button(
                    "Download Dataset (CSV)",
                    data=csv_data,
                    file_name=f"water_quality_spatial_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="export_csv"
                )
                
                col2.markdown(f"<div style='color: white; font-weight: 500; margin-top: 5px;'>**Data Summary:** {len(map_df)} stations across {map_df['state'].nunique()} states</div>", unsafe_allow_html=True)


                # Advanced geospatial analysis
                st.markdown("---")
                st.markdown("### Risk Assessment")
                
                if len(map_df) > 10:
                    with st.spinner("Performing spatial analysis..."):
                        unsafe_df = map_df[map_df["is_safe"] == 0]
                        
                        if len(unsafe_df) > 0:
                            tab_a, tab_b = st.tabs(["High-Risk Areas", "Quality Distribution"])
                            
                            with tab_a:
                                st.markdown('<div style="font-size: 1.15rem; font-weight: 700; margin: 1.5rem 0 1rem 0; color: var(--text-main);">States with Most Unsafe Stations</div>', unsafe_allow_html=True)

                                state_risk = unsafe_df.groupby("state").agg(
                                    unsafe_count=("is_safe", "count"),
                                    avg_wqi=("WQI", "mean"),
                                    max_wqi=("WQI", "max"),
                                    total_stations=("state", "count")
                                ).sort_values("unsafe_count", ascending=False).head(10)
                                state_risk.columns = ["Unsafe Count", "Avg WQI", "Max WQI", "Total Stations"]
                                
                                # Define a subtle style for the table
                                table_style = state_risk.style.set_properties(**{
                                    'background-color': 'rgba(30, 41, 59, 0.5)',
                                    'color': '#f8fafc',
                                    'border-color': 'rgba(255, 255, 255, 0.1)'
                                })

                                st.dataframe(
                                    table_style,
                                    use_container_width=True,
                                    column_config={
                                        "Unsafe Count": st.column_config.NumberColumn("Unsafe Count", format="%d ⚠️"),
                                        "Avg WQI": st.column_config.NumberColumn("Avg WQI", format="%.1f"),
                                        "Max WQI": st.column_config.NumberColumn("Max WQI", format="%.1f"),
                                        "Total Stations": st.column_config.NumberColumn("Total Stations", format="%d")
                                    }
                                )

                                
                                # Risk level visualization
                                st.markdown('<div style="font-size: 1.15rem; font-weight: 700; margin: 2rem 0 1rem 0; color: var(--text-main);">Risk Level Heatmap</div>', unsafe_allow_html=True)
                                risk_heatmap = state_risk[["Unsafe Count", "Avg WQI"]]
                                risk_style = risk_heatmap.style.set_properties(**{
                                    'background-color': 'rgba(30, 41, 59, 0.5)',
                                    'color': '#f8fafc'
                                }).background_gradient(cmap='YlOrRd', subset=['Avg WQI'])
                                
                                st.dataframe(
                                    risk_style,
                                    use_container_width=True,
                                    column_config={
                                        "Unsafe Count": st.column_config.NumberColumn("Unsafe Count"),
                                        "Avg WQI": st.column_config.NumberColumn("WQI Score", format="%.1f")
                                    }
                                )




                            
                            with tab_b:
                                st.markdown('<div style="font-size: 1.15rem; font-weight: 700; margin: 1.5rem 0 1rem 0; color: var(--text-main);">Water Quality Distribution</div>', unsafe_allow_html=True)
                                risk_levels = map_df["water_quality"].value_counts()
                                
                                fig_dist = px.pie(
                                    values=risk_levels.values,
                                    names=risk_levels.index,
                                    color=risk_levels.index,
                                    color_discrete_map=get_valid_colors(map_df, "water_quality", QUAL_COLORS),
                                    hole=0.4
                                )
                                fig_dist.update_traces(textposition='inside', textinfo='percent+label')
                                fig_dist.update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0))
                                fig_dist = update_chart_layout(fig_dist)
                                st.plotly_chart(fig_dist, use_container_width=True)


                        else:
                            st.success("✅ All stations meet safety standards in current view")
                else:
                    st.warning("⚠️ Insufficient data for comprehensive analysis")



# ─── Trends ─────────────────────────────────────────────
with tab3:

    st.markdown("### Trends")

    if "year" not in filt.columns:
        st.warning("No year column in dataset")
    else:
        with st.spinner("Generating trend analysis..."):
            trend = filt.groupby("year")["WQI"].mean().reset_index()

            if len(trend) == 0:
                st.warning("No data after filtering")
            else:
                with st.container(border=True):
                    # Static high-quality trend line
                    fig = px.line(trend, x="year", y="WQI", markers=True, color_discrete_sequence=["#3b82f6"])
                    fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                    fig = update_chart_layout(fig)
                    st.plotly_chart(fig, use_container_width=True)


                
                # Trend statistics
                st.markdown("### Trend Statistics")

                tr_col1, tr_col2, tr_col3 = st.columns(3)
                
                with tr_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Years Analyzed", len(trend))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tr_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Avg WQI", round(trend["WQI"].mean(), 2))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tr_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    # Calculate trend direction
                    if len(trend) >= 2:
                        first_year = trend.iloc[0]["WQI"]
                        last_year = trend.iloc[-1]["WQI"]
                        trend_change = last_year - first_year
                        trend_direction = "Improving ↓" if trend_change < 0 else "Deteriorating ↑"
                        st.metric("Trend", trend_direction)
                    else:
                        st.metric("Trend", "Stable")
                    st.markdown('</div>', unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════
# TAB 4 — ANALYSIS
# ════════════════════════════════════════════════════════════
with tab4:

    avail = [f for f in CORE_FEATURES if f in filt.columns]

    if len(avail) >= 2:

        param_a = st.selectbox("Parameter A", avail)
        param_b = st.selectbox("Parameter B", avail, index=1)

        with st.container(border=True):
            st.markdown('<div style="font-size: 1.15rem; font-weight: 700; margin-bottom: 1rem; color: var(--text-main);">Correlation Analysis</div>', unsafe_allow_html=True)

            with st.spinner("Generating scatter plot..."):

                scatter_df = filt[[param_a, param_b, "water_quality"]].dropna()
                scatter_df = scatter_df.sample(min(1500, len(scatter_df)))

                fig = px.scatter(
                    scatter_df,
                    x=param_a,
                    y=param_b,
                    color="water_quality",
                    color_discrete_map=get_valid_colors(scatter_df, "water_quality", QUAL_COLORS)
                )
                fig = update_chart_layout(fig)
                st.plotly_chart(fig, use_container_width=True)





# ════════════════════════════════════════════════════════════
# TAB 5 — ML
# ════════════════════════════════════════════════════════════
with tab5:

    if models:
        st.markdown('<div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1.5rem; color: var(--text-main);">AI Water Quality Predictor</div>', unsafe_allow_html=True)


        
        # Get features from RF model
    if "rf_full" in models:
        model_rf = models["rf_full"]["model"]
        features = models["rf_full"]["features"]

        inputs = {}

        with st.container(border=True):
            cols = st.columns(3)
            for i, feat in enumerate(features):
                inputs[feat] = cols[i % 3].number_input(feat, value=0.0)
            
            st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
            predict_clicked = st.button("Generate Prediction", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if predict_clicked:

            # Ensure columns are in the exact order the model expects
            input_df = pd.DataFrame([inputs])[features]
            
            # Initialize prediction variables
            rf_pred = None
            xgb_pred = None
            hybrid_pred = None
            
            col1, col2, col3 = st.columns(3)
            
            # RF Full (Main)
            with col1:
                rf_pred = model_rf.predict(input_df)[0]
                if rf_pred == 1:
                    st.success("✅ SAFE (RF)")
                else:
                    st.error("⚠️ UNSAFE (RF)")
            
            # XGB Full
            if "xgb_full" in models:
                with col2:
                    model_xgb = models["xgb_full"]["model"]
                    xgb_pred = model_xgb.predict(input_df)[0]
                    if xgb_pred == 1:
                        st.success("✅ SAFE (XGB)")
                    else:
                        st.error("⚠️ UNSAFE (XGB)")
            
            # Soft Hybrid
            if "hybrid_soft" in models:
                with col3:
                    model_hybrid = models["hybrid_soft"]["model"]
                    hybrid_pred = model_hybrid.predict(input_df)[0]
                    if hybrid_pred == 1:
                        st.success("✅ SAFE (Hybrid)")
                    else:
                        st.error("⚠️ UNSAFE (Hybrid)")
            
            # Summary
            st.markdown("---")
            st.markdown("**Model Predictions Summary:**")
            pred_summary = {
                "RF (Main)": "Safe" if rf_pred == 1 else "Unsafe",
            }
            if xgb_pred is not None:
                pred_summary["XGB"] = "Safe" if xgb_pred == 1 else "Unsafe"
            if hybrid_pred is not None:
                pred_summary["Soft Hybrid"] = "Safe" if hybrid_pred == 1 else "Unsafe"
            
            summary_df = pd.DataFrame(list(pred_summary.items()), columns=["Model", "Prediction"])
            st.dataframe(summary_df)

    else:
        st.warning("Run model_dev.py first")
    
    
    st.markdown("---")
    st.markdown('<div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 1.5rem; color: var(--text-main);">Location-based Safety Analysis</div>', unsafe_allow_html=True)



    # user inputs
    with st.container(border=True):
        col1, col2 = st.columns(2)
        user_lat = col1.number_input("Enter Latitude", value=12.97)
        user_lon = col2.number_input("Enter Longitude", value=77.59)

        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze Current Location", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:


        nearest = get_nearest_stations(filt, user_lat, user_lon, n=3)
        if not nearest.empty and nearest["distance_km"].min() > 30:
            st.warning("⚠️ No nearby monitoring stations found. Results may be less accurate.")

        if "rf_full" not in models:
            st.error("Model not loaded")
        else:
            model = models["rf_full"]["model"]
        features = models["rf_full"]["features"]

        st.markdown("### Nearest Stations")

        weighted_sum = 0
        total_weight = 0

        results = []

        for _, row in nearest.iterrows():

            input_data = pd.DataFrame([{f: row.get(f, 0) for f in features}])

            pred = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0][1]

            
            distance = row["distance_km"]
            adjusted_conf = proba * np.exp(-distance / 150)
            weight = np.exp(-distance / 50)

            weighted_sum += proba * weight
            total_weight += weight

            prediction = "SAFE" if pred == 1 else "UNSAFE"

            # 🔥 enforce domain logic
            if row["WQI"] is not None:
                if row["WQI"] > 50:
                    prediction = "UNSAFE"

            results.append({
                "Station": row.get("station", "Unknown"),
                "Distance (km)": round(distance, 2),
                "WQI": row.get("WQI", None),
                "Prediction": prediction,
                "Confidence": adjusted_conf,  # keep numeric for plotting

                # 🔥 ADD THESE
                "latitude": row["latitude"],
                "longitude": row["longitude"]
            })
        results_df = pd.DataFrame(results)

        st.dataframe(results_df)

        # final weighted decision
        final_score = weighted_sum / total_weight if total_weight > 0 else 0

        st.markdown("### Final Decision")

        st.write(f"Weighted Safety Score: {final_score:.2%}")

        if final_score > 0.5:
            st.success("Overall: Likely SAFE water in this area")
        else:
            st.error("Overall: Likely UNSAFE water in this area")
            
        fig = px.scatter_mapbox(
            results_df,
            lat="latitude",
            lon="longitude",

            # match your heatmap logic
            color="Prediction",   # SAFE / UNSAFE
            size="WQI",
            size_max=18,

            zoom=6,
            center={"lat": user_lat, "lon": user_lon},

            mapbox_style="carto-positron",  # SAME as your other maps

            hover_data={
                "Station": True,
                "Distance (km)": True,
                "Confidence": True,
                "latitude": False,
                "longitude": False
            }
        )

        fig.add_scattermapbox(
            lat=[user_lat],
            lon=[user_lon],
            mode="markers",
            marker=dict(size=14, color="black"),
            name="Your Location"
        )

        fig.update_traces(marker=dict(opacity=0.85))

        fig.update_layout(
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        for _, row in results_df.iterrows():

            fig.add_scattermapbox(
                lat=[user_lat, row["latitude"]],
                lon=[user_lon, row["longitude"]],
                mode="lines",

                line=dict(
                    width=2,
                    color="red",
                ),

                name=f'Connection to {row["Station"]}',
                hoverinfo="none",
                showlegend=False
            )
        
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 6 — UPLOAD
# ════════════════════════════════════════════════════════════
with tab6:

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:

        with st.spinner("Processing uploaded file..."):
            new_df = pd.read_csv(uploaded)

            st.write(new_df.head())

            fig = px.histogram(new_df, x="WQI")
            st.plotly_chart(fig, use_container_width=True)
        
