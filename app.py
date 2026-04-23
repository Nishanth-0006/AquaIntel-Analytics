import os, sys, warnings, joblib
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

# ─── Page config ─────────────────────────────────────────────
# sets layout + theme
st.set_page_config(
    page_title="AquaIntel Analytics",
    page_icon="💧",
    layout="wide",
)

# ─── Custom UI & Interactive Background ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --primary: #27ae60;
    --bg-main: #f8fafc;
    --bg-card: #ffffff;
    --text-main: #1e293b;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --sidebar: #ffffff;
    --glass: rgba(255, 255, 255, 0.7);
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-main: #0f172a;
        --bg-card: #1e293b;
        --text-main: #f8fafc;
        --text-muted: #94a3b8;
        --border: #334155;
        --sidebar: #1e293b;
        --glass: rgba(15, 23, 42, 0.7);
    }
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-main) !important;
    box-sizing: border-box !important;
}

*, *:before, *:after {
    box-sizing: inherit !important;
}

/* Background & Global Animation */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-main);
    animation: fadeIn 0.8s ease-out;
    overflow-x: hidden;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* SaaS Elite: Clean, High-Contrast UI */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #eef2f6 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08) !important;
    transform: translateY(-2px);
}

.card-header-pro {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.icon-box {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.badge-pro {
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    margin-left: auto;
}

.stMetric {
    background: transparent !important;
}

[data-testid="stMetricValue"] > div {
    font-size: 28px !important;
    font-weight: 800 !important;
    color: #1e293b !important;
}

[data-testid="stMetricLabel"] > div {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #64748b !important;
}

/* Water Ripple Animation for Header */
.stTitle {
    background: linear-gradient(-45deg, #27ae60, #2ecc71, #3498db, #2980b9);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Water Ripple Animation for Header */
.stTitle {
    background: linear-gradient(-45deg, #27ae60, #2ecc71, #3498db, #2980b9);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Sidebar "Filter Dashboard" Glassmorphism */
[data-testid="stSidebar"] {
    background: var(--glass) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stMultiSelect"]),
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stSlider"]),
[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.stTextInput) {
    background-color: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(5px);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px 18px 8px 18px;
    margin-bottom: 20px;
}

/* Leaderboard Cards */
.leaderboard-card {
    background: var(--bg-main);
    padding: 12px 16px;
    border-radius: 12px;
    margin-bottom: 8px;
    transition: all 0.3s ease;
    border: 1px solid var(--border);
}
.leaderboard-card:hover {
    background: var(--bg-card);
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    transform: translateX(4px);
}

/* Forced Green Theme for Streamlit Widgets */
span[data-baseweb="tag"] {
    background-color: var(--primary) !important;
}
div[data-testid="stThumb"] {
    background-color: var(--primary) !important;
    border: 2px solid var(--bg-card) !important;
}
div[data-testid="stSlider"] [style*="background-color: rgb(255, 75, 75)"] {
    background-color: var(--primary) !important;
}

/* Tabs & Containers */
div[role="tablist"] {
    background-color: var(--bg-main) !important;
    padding: 6px;
    border-radius: 14px;
}
div[role="tab"][aria-selected="true"] {
    background-color: var(--bg-card) !important;
    color: var(--primary) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    font-weight: 600 !important;
}

/* Plotly Responsive Fix */
.js-plotly-plot, .plot-container {
    width: 100% !important;
}

</style>

""", unsafe_allow_html=True)

# ─── Color mapping ───────────────────────────────────────────
# base colors for water quality categories
QUAL_COLORS = {
    "Excellent": "#27ae60", # Green
    "Good":      "#f1c40f", # Yellow
    "Poor":      "#e67e22", # Orange
    "Very Poor": "#e74c3c", # Red
}

# returns only colors present in dataset
def get_valid_colors(df, column, color_map):
    present = df[column].dropna().astype(str).str.strip().unique()
    return {k: v for k, v in color_map.items() if k in present}


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

# ─── Sidebar ─────────────────────────────────────────────────
st.sidebar.title("🔍 Filter Dashboard")
st.sidebar.markdown("---")

# Search Box (Mockup Style)
search_query = st.sidebar.text_input("Search", placeholder="Search Location or State...", label_visibility="collapsed")

# Initialize session state for filters (Must be at the very top)
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

# Quick Actions (Must be before widgets to avoid locked state)
st.sidebar.markdown("### ⚡ Quick Actions")
qa_col1, qa_col2 = st.sidebar.columns(2)

if qa_col1.button("🔥 Critical", use_container_width=True):
    st.session_state.wqi_filter = (50.0, 100.0)
    st.rerun()

if qa_col2.button("🌿 Safe Only", use_container_width=True):
    st.session_state.wqi_filter = (0.0, 30.0)
    st.rerun()


if st.sidebar.button("🔄 Reset Dashboard", use_container_width=True):
    st.session_state.state_filter = states[:10]
    st.session_state.wqi_filter = (0.0, 100.0)
    if "year" in df.columns:
        yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
        st.session_state.year_filter = (yr_min, yr_max)
    st.rerun()

st.sidebar.markdown("---")

# Location Group
st.sidebar.markdown("##### 📍 Location")
states = sorted(df["state"].dropna().unique())
sel_states = st.sidebar.multiselect(
    "Select States",
    states,
    key="state_filter"
)

# Time Period Group
st.sidebar.markdown("##### 📅 Date Range")
if "year" in df.columns:
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    sel_years = st.sidebar.slider("Select Year Range", yr_min, yr_max, key="year_filter")
else:
    sel_years = None

# Metrics Group
st.sidebar.markdown("##### 📉 Parameters")
wqi_range = st.sidebar.slider("WQI Score Range", 0.0, 100.0, key="wqi_filter")

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

# ─── Header ─────────────────────────────────────────────────
st.title("💧 AquaIntel Analytics")
st.markdown(f"**{len(filt):,} records** · Source: {source}")


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

    st.markdown("### 📊 Executive Summary")
    
    # Smart Status Verdict
    safe_pct_global = (filt["is_safe"].mean() * 100) if "is_safe" in filt.columns and not filt.empty else 0.0
    if not filt.empty:
        if safe_pct_global > 80:
            st.success(f"🌟 **System Status: EXCELLENT** ({safe_pct_global:.1f}% Safe). Water quality across selected regions is optimal.")
        elif safe_pct_global > 50:
            st.warning(f"⚠️ **System Status: CONCERNING** ({safe_pct_global:.1f}% Safe). Some regions require immediate monitoring.")
        else:
            st.error(f"🚨 **System Status: CRITICAL** ({safe_pct_global:.1f}% Safe). Urgent intervention required in high-risk zones.")
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    # Pre-calculate values for cards
    mean_wqi = filt["WQI"].mean() if not filt.empty else 0
    total_recs = len(filt)
    active_states = filt['state'].nunique() if not filt.empty else 0
    safe_pct_global = (filt["is_safe"].mean() * 100) if "is_safe" in filt.columns and not filt.empty else 0.0


    # 1. Overall WQI Score
    with kpi_col1:
        with st.container():
            st.markdown("""
                <div class='card-header-pro'>
                    <div class='icon-box' style='background: #e1f5fe; color: #039be5;'>📊</div>
                    <div style='font-size: 13px; font-weight: 600; color: #64748b;'>Overall WQI Score</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 28px; font-weight: 800; color: #1e293b;'>{mean_wqi:.1f}</div>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge",
                value = mean_wqi,
                gauge = {
                    'axis': {'range': [None, 100], 'visible': False},
                    'bar': {'color': "#27ae60", 'thickness': 0.8},
                    'bgcolor': "#f1f5f9",
                    'borderwidth': 0,
                }
            ))
            fig_gauge.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    # 2. Total Records
    with kpi_col2:
        with st.container():
            st.markdown("""
                <div class='card-header-pro'>
                    <div class='icon-box' style='background: #f0fdf4; color: #16a34a;'>📑</div>
                    <div style='font-size: 13px; font-weight: 600; color: #64748b;'>Total Records</div>
                    <div class='badge-pro' style='background: #f0fdf4; color: #16a34a;'>▲ 85%</div>
                </div>
            """, unsafe_allow_html=True)
            st.metric("Total Records", f"{total_recs:,}", label_visibility="collapsed")
            if "year" in filt.columns and not filt.empty:
                spark_data = filt.groupby("year").size()
                st.plotly_chart(create_sparkline(spark_data), use_container_width=True, config={'displayModeBar': False})

    # 3. Active States
    with kpi_col3:
        with st.container():
            st.markdown("""
                <div class='card-header-pro'>
                    <div class='icon-box' style='background: #fdf2f8; color: #db2777;'>📍</div>
                    <div style='font-size: 13px; font-weight: 600; color: #64748b;'>Active States</div>
                    <div class='badge-pro' style='background: #f0fdf4; color: #16a34a;'>▲ 100%</div>
                </div>
            """, unsafe_allow_html=True)
            st.metric("Active States", active_states, label_visibility="collapsed")
            if "year" in filt.columns and not filt.empty:
                spark_data = filt.groupby("year")["state"].nunique()
                st.plotly_chart(create_sparkline(spark_data, color="#3498db"), use_container_width=True, config={'displayModeBar': False})

    # 4. Safety Index
    with kpi_col4:
        with st.container():
            st.markdown("""
                <div class='card-header-pro'>
                    <div class='icon-box' style='background: #fff1f2; color: #e11d48;'>🛡️</div>
                    <div style='font-size: 13px; font-weight: 600; color: #64748b;'>Safety Index</div>
                    <div class='badge-pro' style='background: #f0fdf4; color: #16a34a;'>SECURE</div>
                </div>
            """, unsafe_allow_html=True)
            st.metric("Safety Index", f"{safe_pct_global:.1f}%", label_visibility="collapsed")
            if "year" in filt.columns and not filt.empty:
                spark_data = filt.groupby("year")["is_safe"].mean() * 100
                st.plotly_chart(create_sparkline(spark_data, color="#e74c3c"), use_container_width=True, config={'displayModeBar': False})


    st.markdown("---")

    # --- Charts Layout ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Quality Distribution")
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
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with legend_col:
                st.markdown("<div style='font-weight: 600; font-size: 14px; margin-bottom: 15px; color: var(--text-main);'>Category Breakdown</div>", unsafe_allow_html=True)
                
                # Build custom HTML legend with progress bars
                legend_html = "<div style='display: flex; flex-direction: column; gap: 12px;'>"
                valid_colors = get_valid_colors(wq_counts, "Category", QUAL_COLORS)
                
                for _, row in wq_counts.iterrows():
                    cat = row["Category"]
                    count = row["Count"]
                    pct = (count / total_count) * 100 if total_count > 0 else 0
                    color = valid_colors.get(cat, "#cccccc")
                    
                    legend_html += f"""
<div style='display: flex; flex-direction: column; gap: 4px;'>
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
</div>
"""
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)

    with chart_col2:
        st.markdown("#### 🏆 Regional Insights")
        
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
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No data available for leaderboard analysis.")





## ─── Heatmap ─────────────────────────────────────────────
with tab2:

    st.markdown("### Water Quality Spatial Analysis")

    if "latitude" in filt.columns:

        # Map configuration - merged controls
        with st.expander("Map Configuration", expanded=True):
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
                    st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False})

                # Enhanced statistics with visual indicators
                st.markdown("---")
                st.markdown("### Spatial Analytics Dashboard")
                
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
                
                col2.markdown(f"**Data Summary:** {len(map_df)} stations across {map_df['state'].nunique()} states")

                # Advanced geospatial analysis
                st.markdown("---")
                st.markdown("### Risk Assessment")
                
                if len(map_df) > 10:
                    with st.spinner("Performing spatial analysis..."):
                        unsafe_df = map_df[map_df["is_safe"] == 0]
                        
                        if len(unsafe_df) > 0:
                            tab_a, tab_b = st.tabs(["High-Risk Areas", "Quality Distribution"])
                            
                            with tab_a:
                                st.markdown("#### States with Most Unsafe Stations")
                                state_risk = unsafe_df.groupby("state").agg(
                                    unsafe_count=("is_safe", "count"),
                                    avg_wqi=("WQI", "mean"),
                                    max_wqi=("WQI", "max"),
                                    total_stations=("state", "count")
                                ).sort_values("unsafe_count", ascending=False).head(10)
                                state_risk.columns = ["Unsafe Count", "Avg WQI", "Max WQI", "Total Stations"]
                                st.dataframe(state_risk, use_container_width=True)
                                
                                # Risk level visualization
                                st.markdown("#### Risk Level Heatmap")
                                risk_heatmap = state_risk[["Unsafe Count", "Avg WQI"]]
                                st.dataframe(risk_heatmap, use_container_width=True)
                            
                            with tab_b:
                                st.markdown("#### Water Quality Distribution")
                                risk_levels = map_df["water_quality"].value_counts()
                                
                                fig_dist = px.pie(
                                    values=risk_levels.values,
                                    names=risk_levels.index,
                                    title="Quality Category Distribution",
                                    hole=0.4
                                )
                                fig_dist.update_traces(textposition='inside', textinfo='percent+label')
                                fig_dist.update_layout(height=400)
                                st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
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
                # Static high-quality trend line
                fig = px.line(trend, x="year", y="WQI", markers=True, color_discrete_sequence=["#27ae60"])
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Trend statistics
                st.markdown("### 📈 Trend Statistics")
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
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — ML
# ════════════════════════════════════════════════════════════
with tab5:

    if models:

        st.markdown("### Water Quality Prediction")
        
        # Get features from RF model
    if "rf_full" in models:
        model_rf = models["rf_full"]["model"]
        features = models["rf_full"]["features"]

        inputs = {}

        cols = st.columns(3)

        for i, feat in enumerate(features):
            inputs[feat] = cols[i % 3].number_input(feat, value=0.0)

        if st.button("Predict"):
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
    st.markdown("## 📍 Location-based Water Safety")

    # user inputs
    user_lat = st.number_input("Enter Latitude", value=12.97)
    user_lon = st.number_input("Enter Longitude", value=77.59)

    if st.button("Analyze Location"):
        st.session_state.run_analysis = True

    if st.session_state.get("run_analysis"):

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
        
