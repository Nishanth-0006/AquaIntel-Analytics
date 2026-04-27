# ═══════════════════════════════════════════════════════════════════════════════
# AquaIntel Analytics — app.py
# CWC Water Quality Monitoring Dashboard | BIS 10500:2012 | SDG Goal 6
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, warnings, joblib, json, re
from math import radians, sin, cos, sqrt, atan2

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    load_all_csvs, generate_synthetic_cwc, preprocess,
    CORE_FEATURES, BIS_STANDARDS,
)
from utils.model_utils import SoftVotingHybrid

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaIntel Analytics — CWC Water Quality",
    page_icon="💧",
    layout="wide",
)

# ─── SECTION: Global CSS (Light Professional Government Theme) ──────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    font-size: 15px !important;
}

/* Remove default top padding */
.stApp > header { display: none !important; }
.block-container { padding-top: 1rem !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1e3a5f !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] strong {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background-color: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: rgba(30,64,175,0.4) !important;
    color: #bfdbfe !important;
    border: 1px solid rgba(96,165,250,0.3) !important;
    border-radius: 20px !important;
}
[data-testid="stSidebar"] .stExpander {
    background-color: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stExpander summary p { color: #93c5fd !important; }

/* Reset button in sidebar — red */
[data-testid="stSidebar"] .stButton > button {
    background-color: #dc2626 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #b91c1c !important;
}

/* ── Main content headings & text ────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }
.stMarkdown p { color: #475569 !important; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
div[role="tablist"] {
    background: #ffffff !important;
    border-bottom: 2px solid #e2e8f0 !important;
    border-radius: 8px 8px 0 0 !important;
}
button[role="tab"] {
    background: transparent !important;
    border: none !important;
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.75rem 1.2rem !important;
}
button[role="tab"][aria-selected="true"] {
    color: #1e40af !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #1e40af !important;
}
button[role="tab"]:hover { color: #1e40af !important; background: #f8fafc !important; }

/* ── Primary Buttons ─────────────────────────────────────────────────────── */
.stButton > button {
    background-color: #1e40af !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(30,64,175,0.3) !important;
}
.stDownloadButton > button, [data-testid="stDownloadButton"] > button {
    background-color: #16a34a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ── Cards / bordered containers ─────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
[data-testid="stDataFrame"] { border-radius: 8px !important; }
[data-testid="stMetricValue"] { color: #0f172a !important; }
[data-testid="stMetricLabel"] { color: #475569 !important; }

/* ── Expanders ───────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #475569 !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
}

/* ── File uploader & alerts ──────────────────────────────────────────────── */
[data-testid="stFileUploadDropzone"] {
    background-color: #f8fafc !important;
    border: 2px dashed #cbd5e1 !important;
}
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ─── SECTION: Constants ─────────────────────────────────────────────────────────
QUAL_COLORS = {
    "Excellent": "#16a34a",
    "Good":      "#2563eb",
    "Poor":      "#d97706",
    "Very Poor": "#dc2626",
}

PARAM_LABELS = {
    "dissolved_oxygen":  "Dissolved Oxygen (mg/L)",
    "total_hardness":    "Total Hardness (mg/L as CaCO₃)",
    "BOD":               "Biochemical Oxygen Demand / BOD (mg/L)",
    "COD":               "Chemical Oxygen Demand / COD (mg/L)",
    "TDS":               "Total Dissolved Solids / TDS (mg/L)",
    "TSS":               "Total Suspended Solids / TSS (mg/L)",
    "nitrates":          "Nitrates (mg/L)",
    "ammonia":           "Ammonia (mg/L)",
    "phosphate":         "Phosphate (mg/L)",
    "chloride":          "Chloride (mg/L)",
    "fluoride":          "Fluoride (mg/L)",
    "sulphate":          "Sulphate (mg/L)",
    "conductivity":      "Electrical Conductivity (µS/cm)",
    "turbidity":         "Turbidity (NTU)",
    "total_coliform":    "Total Coliform (MPN/100mL)",
    "fecal_coliform":    "Fecal Coliform (MPN/100mL)",
    "arsenic":           "Arsenic (mg/L)",
    "lead":              "Lead (mg/L)",
    "iron":              "Iron (mg/L)",
    "manganese":         "Manganese (mg/L)",
    "pH":                "pH",
    "temperature":       "Temperature (°C)",
    "nitrites":          "Nitrites (mg/L)",
    "calcium":           "Calcium (mg/L)",
    "magnesium":         "Magnesium (mg/L)",
}

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─── SECTION: Helper Functions ──────────────────────────────────────────────────

def update_chart_layout(fig, height=450):
    """Apply consistent light-theme layout to every Plotly chart."""
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#0f172a", family="Inter, sans-serif", size=13),
        title=dict(font=dict(color="#0f172a", size=16), x=0, xanchor="left"),
        legend=dict(font=dict(color="#475569", size=12), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            gridcolor="#e2e8f0", zerolinecolor="#e2e8f0",
            tickfont=dict(color="#475569", size=12),
            title_font=dict(color="#0f172a", size=13),
        ),
        yaxis=dict(
            gridcolor="#e2e8f0", zerolinecolor="#e2e8f0",
            tickfont=dict(color="#475569", size=12),
            title_font=dict(color="#0f172a", size=13),
        ),
        margin=dict(l=20, r=20, t=55, b=20),
        height=height,
    )
    return fig


def get_valid_colors(df, column, color_map):
    present = df[column].dropna().astype(str).str.strip().unique()
    return {k: v for k, v in color_map.items() if k in present}


def pct_exceeds(series, bis_info):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    if "max" in bis_info:
        return (s > bis_info["max"]).mean() * 100
    elif "min" in bis_info:
        return (s < bis_info["min"]).mean() * 100
    return np.nan


def wqi_verdict(wqi):
    if wqi < 25:   return "Excellent — Very Clean",    "#16a34a"
    elif wqi < 50: return "Good — Safe for Drinking",  "#2563eb"
    elif wqi < 75: return "Poor — Needs Treatment",    "#d97706"
    else:          return "Very Poor — Unsafe",         "#dc2626"


def compliance_badge(pct_compliant):
    if pct_compliant >= 80:   return "PASS",    "#16a34a", "#dcfce7"
    elif pct_compliant >= 50: return "CAUTION", "#d97706", "#fef3c7"
    else:                     return "FAIL",    "#dc2626", "#fee2e2"


def safety_status(safe_pct):
    if safe_pct > 80:    return "SAFE",     "#16a34a", "#dcfce7"
    elif safe_pct >= 50: return "CAUTION",  "#d97706", "#fef3c7"
    else:                return "CRITICAL", "#dc2626", "#fee2e2"


def get_nearest_stations(df, lat, lon, n=3):
    """Haversine-based nearest monitoring station finder."""
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    df2 = df.dropna(subset=["latitude", "longitude"]).copy()
    df2["distance_km"] = df2.apply(
        lambda r: haversine(lat, lon, r["latitude"], r["longitude"]), axis=1)
    return df2.nsmallest(n, "distance_km")


def section_header(text):
    """Render an uppercase section divider label."""
    st.markdown(
        f'<div style="font-size:0.72rem;font-weight:700;color:#94a3b8;'
        f'text-transform:uppercase;letter-spacing:0.1em;margin:1.4rem 0 0.6rem 0;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label, value, sub, color="#0f172a", badge=None,
             badge_color="#16a34a", badge_bg="#dcfce7", delta_text="",
             delta_color="#94a3b8"):
    badge_html = (
        f'<span style="background:{badge_bg};color:{badge_color};font-size:0.68rem;'
        f'font-weight:700;padding:2px 9px;border-radius:20px;margin-left:6px;">'
        f'{badge}</span>'
    ) if badge else ""
    delta_html = (
        f'<div style="font-size:0.78rem;color:{delta_color};font-weight:600;margin-top:4px;">'
        f'{delta_text}</div>'
    ) if delta_text else ""
    return f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
            padding:1.2rem 1.4rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);height:100%;">
  <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:0.6rem;">{label}</div>
  <div style="font-size:2.4rem;font-weight:800;color:{color};line-height:1.1;">{value}</div>
  <div style="font-size:0.82rem;color:#64748b;margin-top:0.4rem;">{sub}{badge_html}</div>
  {delta_html}
</div>"""


# ─── SECTION: GeoJSON Helpers ───────────────────────────────────────────────────
GEOJSON_RELATIVE_PATH = os.path.join("data", "geo", "india_districts.geojson")


def normalize_geo_name(value):
    if pd.isna(value): return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def make_district_key(state, district):
    state_key    = normalize_geo_name(state)
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
        st.warning(f"Could not load district GeoJSON: {e}")
        return None
    district_keys = ["district","District","DISTRICT","district_name","DISTRICT_NAME",
        "dist_name","DIST_NAME","dt_name","DT_NAME","dtname","DTNAME","NAME_2","NAME2","name","Name","NAME"]
    state_keys = ["state","State","STATE","state_name","STATE_NAME","st_nm","ST_NM","stname","STNAME","NAME_1","NAME1"]
    for feature in geojson.get("features", []):
        properties = feature.setdefault("properties", {})
        district   = first_property(properties, district_keys)
        state      = first_property(properties, state_keys)
        properties["plotly_key"] = make_district_key(state, district)
    return geojson


@st.cache_data
def build_district_choropleth_frame(frame):
    if frame.empty or "WQI" not in frame.columns:
        return pd.DataFrame(columns=["state", "district", "WQI", "plotly_key"])
    district_col = next((c for c in ["district","District","district_name"] if c in frame.columns), None)
    state_col    = next((c for c in ["state","State","state_name"] if c in frame.columns), None)
    if district_col is None:
        return pd.DataFrame(columns=["state", "district", "WQI", "plotly_key"])
    cols = [district_col, "WQI"]
    if state_col: cols.insert(0, state_col)
    district_df = frame[cols].dropna(subset=[district_col, "WQI"]).copy()
    if district_df.empty:
        return pd.DataFrame(columns=["state", "district", "WQI", "plotly_key"])
    rename_map = {district_col: "district"}
    if state_col: rename_map[state_col] = "state"
    district_df = district_df.rename(columns=rename_map)
    if "state" not in district_df.columns: district_df["state"] = ""
    district_df["district"] = district_df["district"].astype(str).str.strip()
    district_df["state"]    = district_df["state"].astype(str).str.strip()
    district_df = (district_df.groupby(["state","district"], as_index=False)["WQI"]
        .mean().sort_values("WQI", ascending=False))
    district_df["plotly_key"] = district_df.apply(
        lambda row: make_district_key(row["state"], row["district"]), axis=1)
    return district_df


# ─── SECTION: Data & Model Loading ─────────────────────────────────────────────
@st.cache_data
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    try:
        raw = load_all_csvs(data_dir); source = "real"
    except FileNotFoundError:
        st.warning("No data files found. Generating synthetic data for demonstration.")
        raw = generate_synthetic_cwc(n=8000); source = "synthetic"
    except Exception as e:
        st.error(f"Error loading data: {e}. Using synthetic data.")
        raw = generate_synthetic_cwc(n=8000); source = "synthetic"
    return preprocess(raw), source


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
                st.warning(f"Could not load model {f}: {e}")
    return models


df, source = load_data()
models     = load_models()

# ─── SECTION: Sidebar ───────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:1.2rem 0 0.8rem 0;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:1rem;">
  <div style="font-size:1.4rem;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">💧 AquaIntel</div>
  <div style="font-size:0.72rem;color:#93c5fd;margin-top:2px;font-weight:500;letter-spacing:0.05em;">
    GOVERNMENT WATER QUALITY PORTAL
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.info("ℹ️ Filters apply to **Dashboard**, **Historical Trends**, and **Parameter Analysis** tabs only.")

with st.sidebar.expander("ℹ️ What is WQI?"):
    st.markdown("""
**Water Quality Index (WQI)** is a score from **0 to 100**.

**Lower score = Cleaner water.**

| Score   | Category      |
|---------|---------------|
| 0–25    | 🟢 Excellent  |
| 25–50   | 🔵 Good       |
| 50–75   | 🟡 Poor       |
| 75–100  | 🔴 Very Poor  |

*Standard: BIS 10500:2012 (Indian Drinking Water Standard)*
""")

st.sidebar.markdown("---")

# Search
st.sidebar.markdown(
    '<p style="color:#94a3b8;font-size:0.72rem;letter-spacing:0.08em;'
    'font-weight:600;text-transform:uppercase;margin-bottom:4px;">SEARCH</p>',
    unsafe_allow_html=True,
)
search_query = st.sidebar.text_input(
    "Search", placeholder="Location, State...", label_visibility="collapsed")

states = sorted(df["state"].dropna().unique())

# Session state defaults
if "state_filter" not in st.session_state: st.session_state.state_filter = []
if "year_filter"  not in st.session_state:
    yr_min2 = int(df["year"].min()) if "year" in df.columns else 1961
    yr_max2 = int(df["year"].max()) if "year" in df.columns else 2020
    st.session_state.year_filter = (yr_min2, yr_max2)
if "wqi_filter"   not in st.session_state: st.session_state.wqi_filter = (0.0, 100.0)

# Geographical filter
st.sidebar.markdown(
    '<p style="color:#94a3b8;font-size:0.72rem;letter-spacing:0.08em;'
    'font-weight:600;text-transform:uppercase;margin:0.8rem 0 4px 0;">GEOGRAPHICAL FILTER</p>',
    unsafe_allow_html=True,
)
sel_states = st.sidebar.multiselect(
    "States", states, key="state_filter", label_visibility="collapsed")

# Year range
st.sidebar.markdown(
    '<p style="color:#94a3b8;font-size:0.72rem;letter-spacing:0.08em;'
    'font-weight:600;text-transform:uppercase;margin:0.8rem 0 4px 0;">YEAR RANGE</p>',
    unsafe_allow_html=True,
)
if "year" in df.columns:
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    sel_years = st.sidebar.slider(
        "Year Range", yr_min, yr_max, (yr_min, yr_max),
        key="year_filter", label_visibility="collapsed")
else:
    sel_years = None

# WQI range
st.sidebar.markdown(
    '<p style="color:#94a3b8;font-size:0.72rem;letter-spacing:0.08em;'
    'font-weight:600;text-transform:uppercase;margin:0.8rem 0 4px 0;">WQI SCORE RANGE</p>',
    unsafe_allow_html=True,
)
wqi_range = st.sidebar.slider(
    "WQI Range", 0.0, 100.0, (0.0, 100.0),
    key="wqi_filter", label_visibility="collapsed")

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Reset Dashboard", use_container_width=True):
    for k in ["state_filter", "year_filter", "wqi_filter"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# ─── SECTION: Data Filtering ────────────────────────────────────────────────────
filt = df.copy()
if sel_states:
    filt = filt[filt["state"].isin(sel_states)]
if sel_years and "year" in filt.columns:
    filt = filt[(filt["year"] >= sel_years[0]) & (filt["year"] <= sel_years[1])]
filt = filt[(filt["WQI"] >= wqi_range[0]) & (filt["WQI"] <= wqi_range[1])]
filt["water_quality"] = filt["water_quality"].astype(str).str.strip()

# ─── SECTION: App Header ────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="color:#0f172a;font-weight:800;font-size:2rem;margin-bottom:0.25rem;">'
    '💧 AquaIntel Analytics</h1>',
    unsafe_allow_html=True,
)
st.markdown(f"""
<div style="background:#1e40af;color:#ffffff;padding:0.65rem 1.2rem;border-radius:8px;
            font-size:0.82rem;font-weight:500;margin-bottom:1.2rem;letter-spacing:0.01em;">
  AquaIntel Analytics — Water Quality Monitoring System &nbsp;|&nbsp;
  CWC Data: 1961–2020 &nbsp;|&nbsp; Standard: BIS 10500:2012 &nbsp;|&nbsp;
  SDG Goal 6: Clean Water &amp; Sanitation &nbsp;|&nbsp;
  <strong>{len(filt):,}</strong> records loaded &nbsp;·&nbsp; Source: {source}
</div>
""", unsafe_allow_html=True)

# ─── SECTION: Tab Layout ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "🗺️ Risk Heatmap",
    "📈 Historical Trends",
    "🔬 Parameter Analysis",
    "🤖 Predict",
    "📤 Upload Data",
])

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Dashboard Tab
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    river_col   = next((c for c in ["River","river_name"] if c in filt.columns), None)
    station_col = next((c for c in ["station_name","station","Station"] if c in filt.columns), None)

    # ── 3A: System Status Banner ─────────────────────────────────────────────────
    safe_pct_global = (filt["is_safe"].mean() * 100) if ("is_safe" in filt.columns and not filt.empty) else 0.0

    if not filt.empty:
        if safe_pct_global > 80:
            icon, label, bc, tc = "✅", "EXCELLENT",   "#16a34a", "#f0fdf4"
        elif safe_pct_global > 50:
            icon, label, bc, tc = "⚠️", "CONCERNING",  "#d97706", "#fffbeb"
        else:
            icon, label, bc, tc = "🚨", "CRITICAL",    "#dc2626", "#fef2f2"

        st.markdown(f"""
<div style="background:{tc};border-left:6px solid {bc};border-radius:8px;
            padding:1rem 1.5rem;margin-bottom:1.2rem;
            display:flex;justify-content:space-between;align-items:center;">
  <div>
    <div style="font-size:1.3rem;font-weight:800;color:{bc};">{icon} SYSTEM STATUS: {label}</div>
    <div style="color:#475569;font-size:0.9rem;margin-top:4px;">
      {safe_pct_global:.1f}% of monitored stations meet BIS 10500:2012 drinking water standards.
    </div>
  </div>
  <div style="text-align:right;font-size:0.78rem;color:#94a3b8;">
    Last Updated<br>
    <strong style="color:#475569;">{pd.Timestamp.now().strftime('%d %b %Y, %H:%M')}</strong>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 3B: KPI Cards ─────────────────────────────────────────────────────────────
    mean_wqi      = filt["WQI"].mean()       if not filt.empty else 0.0
    safe_count    = int(filt["is_safe"].sum()) if ("is_safe" in filt.columns and not filt.empty) else 0
    unsafe_count  = len(filt) - safe_count
    active_states = filt["state"].nunique()   if not filt.empty else 0
    total_stations = filt[station_col].nunique() if (station_col and not filt.empty) else len(filt)

    year_min_f = int(filt["year"].min()) if ("year" in filt.columns and not filt.empty) else "–"
    year_max_f = int(filt["year"].max()) if ("year" in filt.columns and not filt.empty) else "–"

    all_state_names = sorted(filt["state"].dropna().unique()) if not filt.empty else []
    state_names = ", ".join(all_state_names[:8])
    if len(all_state_names) > 8: state_names += "…"

    # WQI delta: first 10% vs last 10% of years
    wqi_delta_text = ""
    delta_color    = "#94a3b8"
    if "year" in filt.columns and not filt.empty and filt["year"].nunique() > 2:
        all_yrs = sorted(filt["year"].dropna().unique())
        n10 = max(1, len(all_yrs) // 10)
        early_wqi = filt[filt["year"].isin(all_yrs[:n10])]["WQI"].mean()
        late_wqi  = filt[filt["year"].isin(all_yrs[-n10:])]["WQI"].mean()
        delta = late_wqi - early_wqi
        wqi_delta_text = f"{'↑ Worsening' if delta > 0 else '↓ Improving'} ({abs(delta):.1f} pts)"
        delta_color    = "#dc2626" if delta > 0 else "#16a34a"
    else:
        wqi_delta_text = "Insufficient data"

    verdict_text, verdict_color = wqi_verdict(mean_wqi)
    s_label, s_color, s_bg = safety_status(safe_pct_global)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_card(
            "AVERAGE WATER QUALITY SCORE",
            f"{mean_wqi:.1f} / 100",
            verdict_text, verdict_color,
            delta_text=wqi_delta_text, delta_color=delta_color,
        ), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card(
            "TOTAL MONITORING STATIONS",
            f"{total_stations:,}",
            f"Across {active_states} states | {year_min_f}–{year_max_f}",
        ), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card(
            "STATES MONITORED",
            f"{active_states}",
            state_names or "—",
        ), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_card(
            "STATIONS PASSING BIS STANDARDS",
            f"{safe_pct_global:.1f}%",
            f"{safe_count:,} Safe | {unsafe_count:,} Require Attention",
            s_color,
            badge=s_label, badge_color=s_color, badge_bg=s_bg,
        ), unsafe_allow_html=True)

    st.markdown("---")
    section_header("WATER QUALITY OVERVIEW")

    # ── 3D: Quality Distribution + State Safety Table ─────────────────────────────
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        with st.container(border=True):
            st.markdown("**Water Quality Category Distribution**")

            wq_counts = filt["water_quality"].value_counts().reset_index()
            wq_counts.columns = ["Category", "Count"]
            total_count = wq_counts["Count"].sum()

            fig_pie = px.pie(
                wq_counts, names="Category", values="Count", color="Category",
                color_discrete_map=get_valid_colors(wq_counts, "Category", QUAL_COLORS),
                hole=0.45,
                title="Water Quality Category Distribution",
            )
            fig_pie.update_traces(
                textposition="inside", textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
            fig_pie = update_chart_layout(fig_pie, height=340)
            fig_pie.update_layout(showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

            # Legend table
            wqi_ranges = {"Excellent":"0–25","Good":"25–50","Poor":"50–75","Very Poor":"75–100"}
            legend_rows = ""
            for cat in ["Excellent","Good","Poor","Very Poor"]:
                row = wq_counts[wq_counts["Category"]==cat]
                cnt = int(row["Count"].values[0]) if len(row) > 0 else 0
                pct = (cnt / total_count * 100) if total_count > 0 else 0
                c   = QUAL_COLORS.get(cat, "#94a3b8")
                legend_rows += f"""
<tr>
  <td style="padding:4px 8px;">
    <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
                 background:{c};margin-right:6px;"></span>{cat}
  </td>
  <td style="padding:4px 8px;color:#64748b;">{wqi_ranges.get(cat,'')}</td>
  <td style="padding:4px 8px;font-weight:600;">{cnt:,}</td>
  <td style="padding:4px 8px;color:{c};font-weight:700;">{pct:.1f}%</td>
</tr>"""
            st.markdown(f"""
<table style="width:100%;border-collapse:collapse;font-size:0.83rem;">
  <thead><tr style="border-bottom:1px solid #e2e8f0;">
    <th style="padding:4px 8px;text-align:left;color:#64748b;font-weight:600;">Category</th>
    <th style="padding:4px 8px;text-align:left;color:#64748b;font-weight:600;">WQI Range</th>
    <th style="padding:4px 8px;text-align:left;color:#64748b;font-weight:600;">Count</th>
    <th style="padding:4px 8px;text-align:left;color:#64748b;font-weight:600;">Share</th>
  </tr></thead>
  <tbody>{legend_rows}</tbody>
</table>""", unsafe_allow_html=True)

            with st.expander("📋 View Data Table"):
                st.dataframe(wq_counts, use_container_width=True)

    with col_right:
        with st.container(border=True):
            st.markdown("**State-wise Safety Status**")
            if not filt.empty:
                state_summary = filt.groupby("state").agg(
                    Stations=("state", "count"),
                    Safe_pct=("is_safe", lambda x: x.mean() * 100),
                    Avg_WQI=("WQI", "mean"),
                ).reset_index()
                state_summary.columns = ["State","Stations","Safe%","Avg WQI"]
                state_summary["Status"] = state_summary["Safe%"].apply(
                    lambda x: safety_status(x)[0])
                state_summary = state_summary.sort_values("Safe%", ascending=True)
                state_summary["Safe%"]   = state_summary["Safe%"].round(1)
                state_summary["Avg WQI"] = state_summary["Avg WQI"].round(1)
                st.dataframe(
                    state_summary,
                    use_container_width=True,
                    height=420,
                    column_config={
                        "State":    st.column_config.TextColumn("State"),
                        "Stations": st.column_config.NumberColumn("Stations", format="%d"),
                        "Safe%":    st.column_config.ProgressColumn(
                                        "Safe %", min_value=0, max_value=100, format="%.1f%%"),
                        "Avg WQI":  st.column_config.NumberColumn("Avg WQI", format="%.1f"),
                        "Status":   st.column_config.TextColumn("Status"),
                    },
                    hide_index=True,
                )
            else:
                st.info("No data for selected filters.")

    # ── 3E: BIS Compliance Heatmap ────────────────────────────────────────────────
    st.markdown("---")
    section_header("PARAMETER COMPLIANCE DASHBOARD")
    st.markdown("**BIS Standard Compliance by State and Parameter**")
    st.caption("Red = High violation rate | Green = Compliant | Values show % of samples exceeding BIS limits")

    avail_bis = [p for p in BIS_STANDARDS if p in filt.columns]
    if not filt.empty and len(avail_bis) > 0 and filt["state"].nunique() > 0:
        with st.spinner("Building compliance heatmap..."):
            pivot_rows = []
            for state_n, grp in filt.groupby("state"):
                row = {"State": state_n}
                for p in avail_bis:
                    row[PARAM_LABELS.get(p, p)] = pct_exceeds(grp[p], BIS_STANDARDS[p])
                pivot_rows.append(row)
            pivot_df = pd.DataFrame(pivot_rows).set_index("State").dropna(axis=1, how="all")

            if not pivot_df.empty:
                fig_heat = px.imshow(
                    pivot_df,
                    color_continuous_scale="RdYlGn_r",
                    zmin=0, zmax=100,
                    text_auto=".0f",
                    aspect="auto",
                    title="BIS Standard Compliance by State and Parameter (% Exceeding Limit)",
                )
                fig_heat.update_traces(
                    textfont=dict(size=10),
                    hovertemplate="<b>%{y}</b><br>%{x}<br><b>%{z:.1f}%</b> samples exceed BIS limit<extra></extra>",
                )
                fig_heat = update_chart_layout(fig_heat, height=max(400, len(pivot_df) * 28))
                fig_heat.update_layout(
                    coloraxis_colorbar=dict(title="% Exceeding", ticksuffix="%"),
                    xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                with st.expander("📋 View Data Table"):
                    st.dataframe(pivot_df.round(1), use_container_width=True)
    else:
        st.info("Insufficient data for compliance heatmap.")

    # ── 3F: River Bar Chart ───────────────────────────────────────────────────────
    st.markdown("---")
    section_header("RIVER & REGIONAL ANALYSIS")

    if river_col and not filt.empty:
        river_summary = (
            filt.groupby(river_col)["WQI"].mean()
            .reset_index().sort_values("WQI", ascending=False).head(10)
        )
        river_summary.columns = ["River Name","WQI"]

        fig_river = px.bar(
            river_summary, x="WQI", y="River Name", orientation="h",
            color="WQI", color_continuous_scale="RdYlGn_r",
            text=river_summary["WQI"].round(1),
            title="Top Rivers by Water Quality Score (Lower = Cleaner)",
        )
        fig_river.update_traces(textposition="outside", textfont=dict(size=12))
        fig_river.add_vline(x=50, line_dash="dash", line_color="#dc2626",
            annotation_text="BIS Threshold (WQI = 50)", annotation_position="top right",
            annotation_font=dict(color="#dc2626", size=11))
        fig_river = update_chart_layout(fig_river, height=450)
        fig_river.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Water Quality Index (WQI) — Lower is Better",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_river, use_container_width=True)
        with st.expander("📋 View Data Table"):
            st.dataframe(river_summary, use_container_width=True)
    else:
        st.info("No river data available for current filters.")

    # ── 3G: Monthly Seasonal Pattern ──────────────────────────────────────────────
    if "month" in filt.columns and not filt.empty:
        monthly = filt.groupby("month")["WQI"].mean().reset_index()
        monthly["Month"] = monthly["month"].apply(
            lambda m: MONTH_NAMES[int(m)-1] if 1 <= int(m) <= 12 else str(m))
        monthly["Color"] = monthly["WQI"].apply(
            lambda w: "#16a34a" if w < 50 else ("#d97706" if w < 75 else "#dc2626"))
        monthly = monthly.sort_values("month")

        fig_season = go.Figure()
        # Monsoon shading Jun–Sep (index 4.5 → 8.5 on 0-indexed bar chart)
        fig_season.add_vrect(
            x0=4.5, x1=8.5,
            fillcolor="rgba(147,197,253,0.18)", line_width=0,
            annotation_text="Monsoon Season ☁️", annotation_position="top left",
            annotation_font=dict(color="#1d4ed8", size=11),
        )
        fig_season.add_bar(
            x=monthly["Month"], y=monthly["WQI"],
            marker_color=monthly["Color"].tolist(),
            text=monthly["WQI"].round(1), textposition="outside",
            hovertemplate="<b>%{x}</b><br>Avg WQI: %{y:.1f}<extra></extra>",
            name="Monthly Avg WQI",
        )
        fig_season.add_hline(y=50, line_dash="dash", line_color="#dc2626",
            annotation_text="BIS Safety Threshold",
            annotation_font=dict(color="#dc2626", size=11))
        fig_season = update_chart_layout(fig_season, height=420)
        fig_season.update_layout(
            title="Seasonal Water Quality Pattern",
            xaxis_title="Month",
            yaxis_title="Average Water Quality Index (WQI) — Lower is Better",
            showlegend=False,
        )
        st.plotly_chart(fig_season, use_container_width=True)
        st.caption("Average WQI by month across all years. June–September = Monsoon Season. "
                   "Higher WQI during monsoon often indicates runoff contamination.")
        with st.expander("📋 View Data Table"):
            st.dataframe(monthly[["Month","WQI"]].rename(columns={"WQI":"Avg WQI"}).round(2),
                         use_container_width=True)

    # ── 3H: Download Report ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**📥 Download Summary Report**")

    if not filt.empty:
        report_rows = []
        for state_n, grp in filt.groupby("state"):
            avail_p = [p for p in BIS_STANDARDS if p in grp.columns]
            exceedances = {p: pct_exceeds(grp[p], BIS_STANDARDS[p]) for p in avail_p}
            exceedances = {k: v for k, v in exceedances.items() if pd.notna(v)}
            most_viol = max(exceedances, key=exceedances.get) if exceedances else "—"
            most_viol_label = PARAM_LABELS.get(most_viol, most_viol)
            s_count = grp[station_col].nunique() if station_col else len(grp)
            sp = grp["is_safe"].mean() * 100 if "is_safe" in grp.columns else 0
            report_rows.append({
                "State":                  state_n,
                "Stations":               s_count,
                "Safe%":                  round(sp, 1),
                "Avg WQI":                round(grp["WQI"].mean(), 1),
                "Most Violated Parameter": most_viol_label,
                "Status":                 safety_status(sp)[0],
            })
        report_df = pd.DataFrame(report_rows)
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        st.download_button(
            "📥 Download Report (CSV)",
            data=report_df.to_csv(index=False),
            file_name=f"AquaIntel_Report_{today}.csv",
            mime="text/csv",
        )

    # ── 3I: Help & Glossary ────────────────────────────────────────────────────────
    with st.expander("📖 Help & Glossary of Terms"):
        st.markdown("""
**GLOSSARY OF TERMS**

**WQI (Water Quality Index):** A score from 0 to 100. Lower score = cleaner water.
- 0–25: Excellent | 25–50: Good | 50–75: Poor | 75–100: Very Poor

**BIS 10500:2012:** Bureau of Indian Standards specification for drinking water quality. The legal standard for India.

**pH:** Measure of acidity/alkalinity. Safe range: 6.5–8.5

**DO (Dissolved Oxygen):** Oxygen dissolved in water. Higher is better. Minimum safe: 6 mg/L

**BOD (Biochemical Oxygen Demand):** Amount of oxygen bacteria need to break down waste. Lower is better. BIS max: 3 mg/L

**TDS (Total Dissolved Solids):** Total minerals dissolved in water. BIS max: 500 mg/L

**Coliform Bacteria:** Indicator of sewage contamination. Should be zero in drinking water.

**Arsenic / Lead:** Heavy metals — toxic even in small amounts. BIS max: 0.01 mg/L each
""")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Risk Heatmap Tab (preserved as-is)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Water Quality Spatial Analysis")

    if "latitude" in filt.columns:
        with st.container(border=True):
            st.markdown("**Map Configuration**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                basemap_style = st.selectbox(
                    "Basemap", ["OpenStreetMap","Carto Positron","Carto Dark"],
                    key="basemap_selector")
            with col2:
                color_theme = st.selectbox(
                    "Color Theme",
                    ["Green-Yellow-Red","Blue-Purple-Red","Viridis","Plasma","Inferno"],
                    key="color_theme_selector")
            with col3:
                color_mode = st.selectbox(
                    "Station Color",
                    ["Safe/Unsafe","WQI Gradient","Quality Categories"],
                    key="color_mode_selector")
            with col4:
                show_density = st.checkbox("Show Density", value=True, key="density_toggle")

            col1b, col2b, col3b = st.columns(3)
            with col1b: density_radius    = st.slider("Density Radius", 5, 30, 18, key="density_radius")
            with col2b: marker_size       = st.slider("Marker Size", 5, 25, 15, key="marker_size")
            with col3b: enable_animation  = st.checkbox("Enable Time Animation", value=True, key="animation_toggle")

        color_themes = {
            "Green-Yellow-Red": [[0.0,"#27ae60"],[0.5,"#f1c40f"],[1.0,"#e74c3c"]],
            "Blue-Purple-Red":  [[0.0,"#3498db"],[0.5,"#9b59b6"],[1.0,"#e74c3c"]],
            "Viridis":          [[0.0,"#440154"],[0.5,"#21918c"],[1.0,"#fde725"]],
            "Plasma":           [[0.0,"#0d0887"],[0.5,"#cc4778"],[1.0,"#f0f921"]],
            "Inferno":          [[0.0,"#000004"],[0.5,"#b53679"],[1.0,"#fcffa4"]],
        }
        selected_colors = color_themes[color_theme]
        basemap_map = {
            "OpenStreetMap":"open-street-map",
            "Carto Positron":"carto-positron",
            "Carto Dark":"carto-darkmatter",
        }
        selected_basemap = basemap_map[basemap_style]

        with st.spinner("Loading spatial data..."):
            map_df = filt.dropna(subset=["latitude","longitude","WQI","water_quality"])
            if len(map_df) == 0:
                st.warning("No valid coordinates found in current filter selection.")
            else:
                if enable_animation and "year" in map_df.columns:
                    available_years  = sorted(map_df["year"].unique())
                    selected_year    = st.selectbox("Select Year", available_years,
                                                    index=len(available_years)-1, key="year_selector")
                    map_df = map_df[map_df["year"] == selected_year]
                    st.info(f"Showing data for year: {selected_year}")

                density_df = map_df.sample(min(2000, len(map_df)), random_state=42)
                scatter_df = map_df.sample(min(1000, len(map_df)), random_state=42)

                if show_density:
                    with st.spinner("Generating density heatmap..."):
                        fig = px.density_mapbox(
                            density_df, lat="latitude", lon="longitude", z="WQI",
                            radius=density_radius, center={"lat":20.5,"lon":80}, zoom=4,
                            mapbox_style="carto-positron",
                            color_continuous_scale=selected_colors,
                            title="Water Quality Density Heatmap",
                            labels={"z":"WQI"},
                            range_color=[density_df["WQI"].min(), density_df["WQI"].max()],
                            opacity=0.8,
                        )
                        fig.update_layout(height=500, margin=dict(l=0,r=0,t=30,b=0),
                            paper_bgcolor="#ffffff",
                            coloraxis_colorbar=dict(title="WQI", x=1.02, len=0.8))
                        st.plotly_chart(fig, use_container_width=True,
                            config={"scrollZoom":True,"displayModeBar":True,"displaylogo":False})

                if color_mode == "Safe/Unsafe":
                    scatter_df = scatter_df.copy()
                    scatter_df["status"] = scatter_df["is_safe"].map({1:"Safe", 0:"Unsafe"})
                    color_col = "status"; color_disc_map = {"Safe":"#27ae60","Unsafe":"#e74c3c"}
                elif color_mode == "WQI Gradient":
                    color_col = "WQI"; color_disc_map = None
                else:
                    color_col = "water_quality"
                    color_disc_map = get_valid_colors(scatter_df, "water_quality", QUAL_COLORS)

                with st.spinner("Rendering station map..."):
                    if color_mode == "WQI Gradient":
                        fig2 = px.scatter_mapbox(
                            scatter_df, lat="latitude", lon="longitude", color="WQI",
                            size="WQI", size_max=marker_size,
                            center={"lat":20.5,"lon":80}, zoom=4, mapbox_style=selected_basemap,
                            title="Water Quality Monitoring Network",
                            color_continuous_scale=selected_colors,
                            hover_data={"WQI":":.2f","water_quality":True,"state":True},
                        )
                    else:
                        fig2 = px.scatter_mapbox(
                            scatter_df, lat="latitude", lon="longitude", color=color_col,
                            color_discrete_map=color_disc_map,
                            size="WQI", size_max=marker_size,
                            center={"lat":20.5,"lon":80}, zoom=4, mapbox_style=selected_basemap,
                            title="Water Quality Monitoring Network",
                            hover_data={"WQI":":.2f","water_quality":True,"state":True},
                        )
                    fig2.update_traces(marker=dict(opacity=0.85))
                    fig2.update_layout(height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="#ffffff")
                    st.plotly_chart(fig2, use_container_width=True,
                        config={"scrollZoom":True,"displayModeBar":True,"displaylogo":False})

    st.markdown("### District-wise Water Quality Map")
    with st.spinner("Building district choropleth..."):
        geojson        = load_india_district_geojson()
        district_map_df = build_district_choropleth_frame(filt)

        if geojson is None:
            st.warning("District GeoJSON missing. Add `data/geo/india_districts.geojson` to enable.")
        elif district_map_df.empty:
            st.warning("No district-level WQI data for current filters.")
        else:
            geojson_keys = {
                f.get("properties",{}).get("plotly_key")
                for f in geojson.get("features",[])
                if f.get("properties",{}).get("plotly_key")
            }
            district_map_df = district_map_df[district_map_df["plotly_key"].isin(geojson_keys)].copy()
            if district_map_df.empty:
                st.warning("District names could not be matched to India district boundaries.")
            else:
                with st.container(border=True):
                    color_scale = [[0.0,"#d7efe4"],[0.35,"#8fcdb2"],[0.65,"#f3d98b"],[1.0,"#de7c6b"]]
                    fig_map = px.choropleth_mapbox(
                        district_map_df, geojson=geojson,
                        locations="plotly_key", featureidkey="properties.plotly_key",
                        color="WQI", color_continuous_scale=color_scale,
                        range_color=(0,100), mapbox_style="carto-positron",
                        zoom=4, center={"lat":20.5,"lon":80},
                        opacity=0.9, custom_data=["district"],
                    )
                    fig_map.update_traces(
                        marker_line_color="white", marker_line_width=0.5,
                        hovertemplate="<b>%{customdata[0]}</b><br>WQI: %{z:.1f}<extra></extra>",
                        showscale=False,
                    )
                    fig_map.update_layout(height=720, margin=dict(l=0,r=0,t=8,b=0),
                        paper_bgcolor="#ffffff", showlegend=False, coloraxis_showscale=False,
                        mapbox=dict(style="carto-positron", zoom=4, center=dict(lat=20.5, lon=80)))
                    st.plotly_chart(fig_map, use_container_width=True,
                        config={"displayModeBar":False,"scrollZoom":True})

    st.markdown("---")
    with st.container(border=True):
        st.markdown("**Spatial Analytics Dashboard**")
        if "latitude" in filt.columns:
            c1, c2, c3, c4 = st.columns(4)
            map_df2 = filt.dropna(subset=["latitude","longitude","WQI","water_quality"])
            safe_pct2 = (len(map_df2[map_df2["is_safe"]==1]) / len(map_df2) * 100) if len(map_df2) > 0 else 0
            c1.metric("Total Stations",   len(map_df2))
            c2.metric("Safe Stations",    len(map_df2[map_df2["is_safe"]==1]), delta=f"{safe_pct2:.1f}%")
            c3.metric("Unsafe Stations",  len(map_df2[map_df2["is_safe"]==0]))
            c4.metric("Mean WQI",         round(map_df2["WQI"].mean(), 2))
            st.markdown("---")
            dl1, dl2 = st.columns(2)
            dl1.download_button(
                "Download Spatial Dataset (CSV)",
                data=map_df2.to_csv(index=False),
                file_name=f"water_quality_spatial_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", key="export_csv",
            )
            dl2.markdown(f"**Data Summary:** {len(map_df2)} stations across {map_df2['state'].nunique()} states")

            if len(map_df2) > 10:
                unsafe_df2 = map_df2[map_df2["is_safe"]==0]
                if len(unsafe_df2) > 0:
                    tab_a, tab_b = st.tabs(["High-Risk Areas","Quality Distribution"])
                    with tab_a:
                        state_risk = (unsafe_df2.groupby("state")
                            .agg(unsafe_count=("is_safe","count"),
                                 avg_wqi=("WQI","mean"),
                                 max_wqi=("WQI","max"))
                            .sort_values("unsafe_count", ascending=False).head(10))
                        state_risk.columns = ["Unsafe Count","Avg WQI","Max WQI"]
                        st.dataframe(state_risk, use_container_width=True, column_config={
                            "Unsafe Count": st.column_config.NumberColumn(format="%d ⚠️"),
                            "Avg WQI":      st.column_config.NumberColumn(format="%.1f"),
                            "Max WQI":      st.column_config.NumberColumn(format="%.1f"),
                        })
                    with tab_b:
                        risk_levels = map_df2["water_quality"].value_counts()
                        fig_dist = px.pie(
                            values=risk_levels.values, names=risk_levels.index,
                            color=risk_levels.index,
                            color_discrete_map=get_valid_colors(map_df2,"water_quality",QUAL_COLORS),
                            hole=0.4, title="Water Quality Distribution",
                        )
                        fig_dist.update_traces(textposition="inside", textinfo="percent+label")
                        fig_dist.update_layout(height=400, margin=dict(t=30,b=0,l=0,r=0),
                            paper_bgcolor="#ffffff")
                        st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Historical Trends Tab
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Historical Water Quality Trends")

    if "year" not in filt.columns:
        st.warning("No year column in dataset.")
    else:
        # ── 4A: Tab-level filters ─────────────────────────────────────────────────
        t3c1, t3c2 = st.columns([1, 1])
        with t3c1:
            tab_states = st.multiselect(
                "Compare States (leave empty for national average)",
                sorted(filt["state"].dropna().unique()),
                default=[], key="trend_states",
            )
        with t3c2:
            yr_min_t, yr_max_t = int(filt["year"].min()), int(filt["year"].max())
            tab_years = st.slider(
                "Year Range (Trends)", yr_min_t, yr_max_t,
                (yr_min_t, yr_max_t), key="trend_years")

        trend_df = filt[(filt["year"] >= tab_years[0]) & (filt["year"] <= tab_years[1])].copy()

        # ── 4B: Main trend line chart ─────────────────────────────────────────────
        with st.container(border=True):
            fig_trend = go.Figure()

            # Green / red zone shading
            fig_trend.add_hrect(y0=0,  y1=50,  fillcolor="rgba(22,163,74,0.06)",  line_width=0)
            fig_trend.add_hrect(y0=50, y1=100, fillcolor="rgba(220,38,38,0.05)", line_width=0)

            colors_line = ["#1e40af","#16a34a","#d97706","#dc2626",
                           "#7c3aed","#0891b2","#be185d","#059669"]

            if tab_states:
                for i, st_name in enumerate(tab_states):
                    grp = trend_df[trend_df["state"]==st_name].groupby("year")["WQI"].mean().reset_index()
                    if not grp.empty:
                        fig_trend.add_scatter(
                            x=grp["year"], y=grp["WQI"], mode="lines+markers",
                            name=st_name,
                            line=dict(color=colors_line[i % len(colors_line)], width=2),
                            marker=dict(size=5),
                            hovertemplate=(
                                f"Year: %{{x}}<br>State: {st_name}<br>"
                                f"Avg WQI: %{{y:.1f}}<br>Status: %{{customdata}}<extra></extra>"
                            ),
                            customdata=["Safe" if v <= 50 else "Unsafe" for v in grp["WQI"]],
                        )
            else:
                national = trend_df.groupby("year")["WQI"].mean().reset_index()
                if not national.empty:
                    fig_trend.add_scatter(
                        x=national["year"], y=national["WQI"], mode="lines+markers",
                        name="National Average",
                        line=dict(color="#1e40af", width=3),
                        marker=dict(size=6),
                        hovertemplate="Year: %{x}<br>National Avg WQI: %{y:.1f}<extra></extra>",
                    )

            fig_trend.add_hline(
                y=50, line_dash="dash", line_color="#dc2626", line_width=2,
                annotation_text="BIS Safety Threshold (WQI = 50)",
                annotation_position="top left",
                annotation_font=dict(color="#dc2626", size=12),
            )
            fig_trend = update_chart_layout(fig_trend, height=480)
            fig_trend.update_layout(
                title="Water Quality Index Trend (1961–2020)",
                xaxis_title="Year",
                yaxis_title="Water Quality Index (WQI) — Lower is Better",
                yaxis=dict(range=[0, 100]),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        national_all = trend_df.groupby("year")["WQI"].mean().reset_index()
        with st.expander("📋 View Data Table"):
            st.dataframe(national_all.round(2), use_container_width=True)

        # ── 4C: Auto-generated plain English insights ─────────────────────────────
        if len(national_all) >= 3:
            x = national_all["year"].values
            y = national_all["WQI"].values
            slope    = np.polyfit(x, y, 1)[0]
            best_yr  = national_all.loc[national_all["WQI"].idxmin()]
            worst_yr = national_all.loc[national_all["WQI"].idxmax()]
            trend_dir  = "worsening" if slope > 0 else "improving"
            trend_icon = "📈" if slope > 0 else "📉"

            state_trends = []
            for sn, sg in trend_df.groupby("state"):
                sg_yr = sg.groupby("year")["WQI"].mean().reset_index()
                if len(sg_yr) >= 3:
                    s2 = np.polyfit(sg_yr["year"].values, sg_yr["WQI"].values, 1)[0]
                    state_trends.append((sn, s2))

            most_improved = min(state_trends, key=lambda x: x[1])[0] if state_trends else "N/A"
            most_declined = max(state_trends, key=lambda x: x[1])[0] if state_trends else "N/A"

            slope_color = "#dc2626" if slope > 0 else "#16a34a"
            st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
            padding:1.2rem 1.5rem;margin:1rem 0;">
  <div style="font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:0.8rem;">
    📊 Key Insights
  </div>
  <ul style="color:#475569;margin:0;padding-left:1.2rem;line-height:2.2;">
    <li><b>Overall trend:</b> Water quality has been
        <b style="color:{slope_color};">{trend_dir}</b>
        since {int(x[0])} (slope: {slope:+.3f} WQI/year) {trend_icon}
    </li>
    <li><b>Best recorded year:</b> {int(best_yr['year'])} — Avg WQI: {best_yr['WQI']:.1f} 🏆</li>
    <li><b>Worst recorded year:</b> {int(worst_yr['year'])} — Avg WQI: {worst_yr['WQI']:.1f} ⚠️</li>
    <li><b>Most improved state:</b> {most_improved} ↓</li>
    <li><b>Most declined state:</b> {most_declined} ↑</li>
  </ul>
</div>""", unsafe_allow_html=True)

        # ── 4D: Decade-wise comparison ────────────────────────────────────────────
        st.markdown("---")
        if "year" in trend_df.columns:
            trend_df2 = trend_df.copy()
            trend_df2["Decade"] = trend_df2["year"].apply(lambda y: f"{(int(y)//10)*10}s")
            decade_avg = (trend_df2.groupby("Decade")["WQI"].mean()
                          .reset_index().sort_values("Decade"))
            decade_avg.columns = ["Decade","Avg WQI"]
            decade_avg["Color"] = decade_avg["Avg WQI"].apply(
                lambda w: "#16a34a" if w < 50 else ("#d97706" if w < 75 else "#dc2626"))

            fig_dec = go.Figure()
            fig_dec.add_bar(
                x=decade_avg["Decade"], y=decade_avg["Avg WQI"],
                marker_color=decade_avg["Color"].tolist(),
                text=decade_avg["Avg WQI"].round(1), textposition="outside",
                hovertemplate="<b>%{x}</b><br>Avg WQI: %{y:.1f}<extra></extra>",
            )
            fig_dec.add_hline(y=50, line_dash="dash", line_color="#dc2626",
                annotation_text="BIS Safety Threshold",
                annotation_font=dict(color="#dc2626"))
            fig_dec = update_chart_layout(fig_dec, height=420)
            fig_dec.update_layout(
                title="Decade-wise Water Quality Comparison",
                xaxis_title="Decade",
                yaxis_title="Average Water Quality Index (WQI)",
                showlegend=False,
            )
            st.plotly_chart(fig_dec, use_container_width=True)
            with st.expander("📋 View Data Table"):
                st.dataframe(decade_avg.round(2), use_container_width=True)

        # ── 4E: State trend ranking table ────────────────────────────────────────
        st.markdown("---")
        st.markdown("**State Performance Over Time**")

        all_yrs_sorted = sorted(trend_df["year"].dropna().unique())
        n_early = max(1, len(all_yrs_sorted) * 2 // 5)
        early_cutoff  = all_yrs_sorted[n_early]     if len(all_yrs_sorted) > n_early else all_yrs_sorted[-1]
        recent_cutoff = all_yrs_sorted[max(0, len(all_yrs_sorted) - max(1, len(all_yrs_sorted)//5))]

        state_perf = []
        for sn, sg in trend_df.groupby("state"):
            early  = sg[sg["year"] <= early_cutoff]["WQI"].mean()
            recent = sg[sg["year"] >= recent_cutoff]["WQI"].mean()
            if pd.notna(early) and pd.notna(recent):
                chg = recent - early
                state_perf.append({
                    "State":             sn,
                    "Early Period WQI":  round(early,  1),
                    "Recent Period WQI": round(recent, 1),
                    "Change":            round(chg,    1),
                    "Trend":             "↑ Worsening" if chg > 0 else "↓ Improving",
                })
        if state_perf:
            perf_df = pd.DataFrame(state_perf).sort_values("Change", ascending=False)
            st.dataframe(perf_df, use_container_width=True, hide_index=True, column_config={
                "State":             st.column_config.TextColumn("State"),
                "Early Period WQI":  st.column_config.NumberColumn(format="%.1f"),
                "Recent Period WQI": st.column_config.NumberColumn(format="%.1f"),
                "Change":            st.column_config.NumberColumn("Change", format="%.1f"),
                "Trend":             st.column_config.TextColumn("Trend"),
            })


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Parameter Analysis Tab
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    avail = [f for f in CORE_FEATURES if f in filt.columns]

    if len(avail) < 2:
        st.warning("Insufficient parameter columns in the dataset for analysis.")
    else:
        avail_labels = [PARAM_LABELS.get(f, f) for f in avail]
        label_to_col = {PARAM_LABELS.get(f, f): f for f in avail}

        # ── 5A: Parameter Health Scorecard ────────────────────────────────────────
        st.markdown("**Parameter-wise BIS Compliance Scorecard**")
        st.caption("Shows what % of water samples meet Indian drinking water standards (BIS 10500:2012)")

        avail_bis2 = [p for p in avail if p in BIS_STANDARDS]
        if avail_bis2 and not filt.empty:
            cols_sc = st.columns(3)
            for i, param in enumerate(avail_bis2):
                pct_exceed  = pct_exceeds(filt[param], BIS_STANDARDS[param])
                compliance  = 100 - pct_exceed if pd.notna(pct_exceed) else 0.0
                badge, bcolor, bbg = compliance_badge(compliance)
                bis_val = BIS_STANDARDS[param]
                bis_str = f"Max: {bis_val.get('max', bis_val.get('min','?'))}"
                label   = PARAM_LABELS.get(param, param)
                with cols_sc[i % 3]:
                    st.markdown(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
            padding:1rem;margin-bottom:0.8rem;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
  <div style="font-size:0.78rem;font-weight:600;color:#0f172a;">{label}</div>
  <div style="font-size:0.7rem;color:#94a3b8;margin:2px 0;">BIS Limit — {bis_str}</div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px;">
    <div style="font-size:1.4rem;font-weight:800;color:{bcolor};">{compliance:.0f}%</div>
    <span style="background:{bbg};color:{bcolor};font-size:0.65rem;font-weight:700;
                 padding:2px 8px;border-radius:20px;">{badge}</span>
  </div>
  <div style="background:#e2e8f0;border-radius:4px;height:6px;margin-top:8px;">
    <div style="background:{bcolor};width:{min(compliance,100):.0f}%;height:100%;border-radius:4px;"></div>
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── 5B: Scatter / Correlation Plot ────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            sel_a_label = st.selectbox("Parameter A (X-axis)", avail_labels, key="param_a")
        with col_b:
            default_b   = min(1, len(avail_labels) - 1)
            sel_b_label = st.selectbox("Parameter B (Y-axis)", avail_labels,
                                       index=default_b, key="param_b")

        param_a = label_to_col.get(sel_a_label, avail[0])
        param_b = label_to_col.get(sel_b_label, avail[min(1, len(avail)-1)])

        with st.container(border=True):
            scatter_d = filt[[param_a, param_b, "water_quality"]].dropna()
            scatter_d = scatter_d.sample(min(1500, len(scatter_d)), random_state=42)

            fig_sc = px.scatter(
                scatter_d, x=param_a, y=param_b, color="water_quality",
                color_discrete_map=get_valid_colors(scatter_d, "water_quality", QUAL_COLORS),
                title=f"Correlation Analysis: {sel_a_label} vs {sel_b_label}",
                labels={param_a: sel_a_label, param_b: sel_b_label},
                opacity=0.65,
                hover_data={"water_quality": True},
            )
            fig_sc = update_chart_layout(fig_sc, height=480)

            # BIS limit reference lines
            bis_val_a = bis_val_b = None
            if param_a in BIS_STANDARDS:
                bis_a   = BIS_STANDARDS[param_a]
                bis_val_a = bis_a.get("max", bis_a.get("min"))
                if bis_val_a:
                    fig_sc.add_vline(x=bis_val_a, line_dash="dash", line_color="#dc2626",
                        annotation_text=f"BIS Limit — {sel_a_label}",
                        annotation_font=dict(color="#dc2626", size=10))
            if param_b in BIS_STANDARDS:
                bis_b   = BIS_STANDARDS[param_b]
                bis_val_b = bis_b.get("max", bis_b.get("min"))
                if bis_val_b:
                    fig_sc.add_hline(y=bis_val_b, line_dash="dash", line_color="#dc2626",
                        annotation_text=f"BIS Limit — {sel_b_label}",
                        annotation_font=dict(color="#dc2626", size=10))

            # Quadrant labels (if both BIS lines exist)
            if bis_val_a and bis_val_b:
                x_vals = scatter_d[param_a].dropna()
                y_vals = scatter_d[param_b].dropna()
                x_mid = x_vals.quantile(0.75)
                y_mid = y_vals.quantile(0.25)
                for qx, qy, qlabel in [
                    (bis_val_a * 1.4, bis_val_b * 1.4, "Both Above BIS Limit"),
                    (bis_val_a * 0.4, bis_val_b * 1.4, "Y Above BIS Limit"),
                    (bis_val_a * 1.4, bis_val_b * 0.4, "X Above BIS Limit"),
                    (bis_val_a * 0.4, bis_val_b * 0.4, "✓ Both Within Safe Range"),
                ]:
                    fig_sc.add_annotation(x=qx, y=qy, text=qlabel,
                        showarrow=False, font=dict(size=9, color="#94a3b8"),
                        bgcolor="rgba(255,255,255,0.7)")

            st.plotly_chart(fig_sc, use_container_width=True)

            # Pearson r interpretation
            try:
                r = np.corrcoef(scatter_d[param_a].values, scatter_d[param_b].values)[0, 1]
                strength  = "strong"   if abs(r) > 0.7 else ("moderate" if abs(r) > 0.4 else "weak")
                direction = "positive" if r > 0 else "negative"
                st.markdown(
                    f"📊 **Statistical Relationship:** {sel_a_label} and {sel_b_label} show "
                    f"**{strength} {direction} correlation** (r = {r:.2f}). "
                    f"{'Higher values of both parameters tend to co-occur.' if r > 0.4 else ('Higher X values associate with lower Y values.' if r < -0.4 else 'No strong linear relationship detected.')}"
                )
            except Exception:
                pass

            with st.expander("📋 View Data Table"):
                st.dataframe(scatter_d.head(500), use_container_width=True)

        # ── 5C: Box Plot — Parameter by State ────────────────────────────────────
        st.markdown("---")
        param_box_label = st.selectbox(
            "Select Parameter for State Distribution", avail_labels, key="box_param")
        param_box = label_to_col.get(param_box_label, avail[0])

        with st.container(border=True):
            box_df = filt[["state", param_box]].dropna(subset=["state", param_box])
            if not box_df.empty:
                state_medians = box_df.groupby("state")[param_box].median()
                bis_limit = None
                if param_box in BIS_STANDARDS:
                    bis_info  = BIS_STANDARDS[param_box]
                    bis_limit = bis_info.get("max", bis_info.get("min"))

                sample_box = box_df.sample(min(3000, len(box_df)), random_state=42)
                fig_box = px.box(
                    sample_box, x="state", y=param_box,
                    title=f"Distribution of {param_box_label} Across States",
                    labels={"state": "State", param_box: param_box_label},
                    points="outliers", color="state",
                    color_discrete_sequence=["#1e40af"],
                )
                if bis_limit is not None:
                    fig_box.add_hline(y=bis_limit, line_dash="dash", line_color="#dc2626",
                        annotation_text=f"BIS Limit: {bis_limit}",
                        annotation_font=dict(color="#dc2626", size=11))
                fig_box = update_chart_layout(fig_box, height=480)
                fig_box.update_layout(showlegend=False, xaxis=dict(tickangle=-35))
                st.plotly_chart(fig_box, use_container_width=True)

                if bis_limit is not None:
                    if param_box == "dissolved_oxygen":
                        safe_states = (state_medians >= bis_limit).sum()
                    else:
                        safe_states = (state_medians <= bis_limit).sum()
                    st.caption(
                        f"**{safe_states} out of {len(state_medians)} states** have median "
                        f"{param_box_label} within safe BIS limits."
                    )

                with st.expander("📋 View Data Table"):
                    st.dataframe(box_df.head(500), use_container_width=True)

        # ── 5D: Exceedance Timeline ────────────────────────────────────────────────
        st.markdown("---")
        param_exc_label = st.selectbox(
            "Select Parameter for Exceedance Timeline", avail_labels, key="exc_param")
        param_exc = label_to_col.get(param_exc_label, avail[0])

        if "year" in filt.columns and param_exc in BIS_STANDARDS:
            with st.container(border=True):
                bis_exc  = BIS_STANDARDS[param_exc]
                exc_rows = []
                for yr, grp in filt.groupby("year"):
                    p = pct_exceeds(grp[param_exc], bis_exc)
                    if pd.notna(p):
                        exc_rows.append({"Year": int(yr), "Exceedance %": round(p, 1)})

                if exc_rows:
                    exc_df = pd.DataFrame(exc_rows).sort_values("Year")
                    fig_exc = go.Figure()
                    fig_exc.add_scatter(
                        x=exc_df["Year"], y=exc_df["Exceedance %"],
                        mode="lines+markers", name="Exceedance %",
                        line=dict(color="#dc2626", width=2), marker=dict(size=5),
                        fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
                    )
                    fig_exc.add_hline(y=20, line_dash="dash", line_color="#d97706",
                        annotation_text="Concern Threshold (20%)",
                        annotation_font=dict(color="#d97706", size=11))
                    fig_exc = update_chart_layout(fig_exc, height=420)
                    fig_exc.update_layout(
                        title=f"{param_exc_label} — Annual BIS Exceedance Rate (%)",
                        xaxis_title="Year",
                        yaxis_title="% Samples Exceeding Safe Limit",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_exc, use_container_width=True)
                    with st.expander("📋 View Data Table"):
                        st.dataframe(exc_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Predict Tab (AI Water Quality Predictor + Location Safety Analysis)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    if models:
        st.markdown(
            '<div style="font-size:1.5rem;font-weight:800;color:#0f172a;margin-bottom:1.5rem;">'
            '🤖 AI Water Quality Predictor</div>',
            unsafe_allow_html=True,
        )

    if "rf_full" in models:
        model_rf = models["rf_full"]["model"]
        features = models["rf_full"]["features"]
        inputs   = {}

        with st.container(border=True):
            cols_p = st.columns(3)
            for i, feat in enumerate(features):
                inputs[feat] = cols_p[i % 3].number_input(
                    PARAM_LABELS.get(feat, feat), value=0.0, key=f"pred_{feat}")
            predict_clicked = st.button("🔍 Generate Prediction", use_container_width=True)

        if predict_clicked:
            input_df = pd.DataFrame([inputs])[features]
            rf_pred = xgb_pred = hybrid_pred = None

            col1p, col2p, col3p = st.columns(3)
            with col1p:
                rf_pred = model_rf.predict(input_df)[0]
                if rf_pred == 1: st.success("✅ SAFE (Random Forest)")
                else:            st.error("⚠️ UNSAFE (Random Forest)")

            if "xgb_full" in models:
                with col2p:
                    xgb_pred = models["xgb_full"]["model"].predict(input_df)[0]
                    if xgb_pred == 1: st.success("✅ SAFE (XGBoost)")
                    else:             st.error("⚠️ UNSAFE (XGBoost)")

            if "hybrid_soft" in models:
                with col3p:
                    hybrid_pred = models["hybrid_soft"]["model"].predict(input_df)[0]
                    if hybrid_pred == 1: st.success("✅ SAFE (Hybrid Ensemble)")
                    else:                st.error("⚠️ UNSAFE (Hybrid Ensemble)")

            st.markdown("---")
            pred_summary = {"RF (Main)": "Safe" if rf_pred == 1 else "Unsafe"}
            if xgb_pred    is not None: pred_summary["XGBoost"]     = "Safe" if xgb_pred    == 1 else "Unsafe"
            if hybrid_pred is not None: pred_summary["Soft Hybrid"]  = "Safe" if hybrid_pred == 1 else "Unsafe"
            st.dataframe(pd.DataFrame(list(pred_summary.items()), columns=["Model","Prediction"]))
    else:
        st.warning("No models found. Run `model_dev.py` first to train and save models.")
                # ─────────────────────────────────────────────
# ⚡ Lite Prediction (3 Parameter Version)
# ─────────────────────────────────────────────
st.markdown("---")

st.markdown(
    '<div style="font-size:1.5rem;font-weight:800;color:#0f172a;margin-bottom:0.5rem;">'
    '⚡ Lite Water Quality Prediction</div>',
    unsafe_allow_html=True,
)

st.caption("Quick estimation using pH, Conductivity, and Nitrates (for demo/educational use)")

col1, col2, col3 = st.columns(3)

ph = col1.number_input("pH", 0.0, 14.0, 7.0)
cond = col2.number_input("Conductivity (µS/cm)", 0.0, 2000.0)
nit = col3.number_input("Nitrates (mg/L)", 0.0, 100.0)
if st.button("⚡ Run Lite Prediction", key="lite_btn", use_container_width=True):

    score = 0

    if ph < 6.5 or ph > 8.5:
        score += 1
    if cond > 750:
        score += 1
    if nit > 45:
        score += 1

    if score == 0:
        st.success("✅ SAFE - Within acceptable limits")
    elif score == 1:
        st.warning("⚠️ MODERATE - Check recommended")
    else:
        st.error("🚨 UNSAFE - Treatment required")


    st.markdown("---")
    st.markdown(
        '<div style="font-size:1.5rem;font-weight:800;color:#0f172a;margin-bottom:1.5rem;">'
        '📍 Location-based Safety Analysis</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        col1l, col2l = st.columns(2)
        user_lat = col1l.number_input("Enter Latitude",  value=12.97)
        user_lon = col2l.number_input("Enter Longitude", value=77.59)
        analyze_clicked = st.button("🔎 Analyze Current Location", use_container_width=True)

    if analyze_clicked:
        nearest = get_nearest_stations(filt, user_lat, user_lon, n=3)

        if not nearest.empty and nearest["distance_km"].min() > 30:
            st.warning("⚠️ No nearby monitoring stations within 30 km. Results may be less accurate.")

        if "rf_full" not in models:
            st.error("Model not loaded. Run model_dev.py first.")
        else:
            model    = models["rf_full"]["model"]
            features = models["rf_full"]["features"]
            st.markdown("### Nearest Monitoring Stations")
            weighted_sum = total_weight = 0
            results      = []

            for _, row in nearest.iterrows():
                input_data = pd.DataFrame([{f: row.get(f, 0) for f in features}])
                pred     = model.predict(input_data)[0]
                proba    = model.predict_proba(input_data)[0][1]
                distance = row["distance_km"]
                adjusted_conf = proba * np.exp(-distance / 150)
                weight        = np.exp(-distance / 50)
                weighted_sum += proba * weight
                total_weight += weight
                prediction = "SAFE" if pred == 1 else "UNSAFE"
                if row.get("WQI") and row["WQI"] > 50:
                    prediction = "UNSAFE"
                results.append({
                    "Station":       row.get("station_name", row.get("station","Unknown")),
                    "Distance (km)": round(distance, 2),
                    "WQI":           row.get("WQI"),
                    "Prediction":    prediction,
                    "Confidence":    round(adjusted_conf, 3),
                    "latitude":      row["latitude"],
                    "longitude":     row["longitude"],
                })

            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            final_score = weighted_sum / total_weight if total_weight > 0 else 0
            st.markdown("### Final Decision")
            st.write(f"Weighted Safety Score: **{final_score:.2%}**")
            if final_score > 0.5:
                st.success("Overall: Water in this area is likely **SAFE**.")
            else:
                st.error("Overall: Water in this area is likely **UNSAFE**. Advise treatment.")

            fig_loc = px.scatter_mapbox(
                results_df, lat="latitude", lon="longitude",
                color="Prediction", size="WQI", size_max=18, zoom=6,
                center={"lat": user_lat, "lon": user_lon},
                mapbox_style="carto-positron",
                hover_data={"Station":True,"Distance (km)":True,
                            "Confidence":True,"latitude":False,"longitude":False},
            )
            fig_loc.add_scattermapbox(lat=[user_lat], lon=[user_lon], mode="markers",
                marker=dict(size=14, color="black"), name="Your Location")
            for _, row in results_df.iterrows():
                fig_loc.add_scattermapbox(
                    lat=[user_lat, row["latitude"]], lon=[user_lon, row["longitude"]],
                    mode="lines", line=dict(width=2, color="red"),
                    name=f'To {row["Station"]}', hoverinfo="none", showlegend=False,
                )
            fig_loc.update_layout(height=500, margin=dict(l=0,r=0,t=30,b=0),
                paper_bgcolor="#ffffff",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_loc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Upload Data Tab
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 📤 Upload CSV Data")
    st.info(
        "Upload a CSV file containing water quality measurements. "
        "Include a **WQI** column for automatic quality visualisation."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        with st.spinner("Processing uploaded file..."):
            new_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(new_df):,} rows × {len(new_df.columns)} columns")
            st.write(new_df.head())

        if "WQI" in new_df.columns:
            fig_up = px.histogram(
                new_df, x="WQI", nbins=40,
                title="WQI Distribution in Uploaded Data",
                labels={"WQI":"Water Quality Index (WQI)"},
                color_discrete_sequence=["#1e40af"],
            )
            fig_up.add_vline(x=50, line_dash="dash", line_color="#dc2626",
                annotation_text="BIS Safety Threshold (WQI = 50)")
            fig_up = update_chart_layout(fig_up, height=420)
            fig_up.update_layout(xaxis_title="Water Quality Index (WQI) — Lower is Better",
                                 yaxis_title="Number of Records")
            st.plotly_chart(fig_up, use_container_width=True)

            # Summary stats
            with st.expander("📋 Summary Statistics"):
                st.dataframe(new_df.describe().round(2), use_container_width=True)
        else:
            st.warning(
                "No 'WQI' column found in uploaded file. "
                "Add a WQI column to enable quality visualisation."
            )
