"""
AquaIntel Analytics — Streamlit Dashboard
Run: streamlit run app.py
"""

import os, sys, warnings, joblib
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data_loader import (
    load_all_csvs, generate_synthetic_cwc, preprocess,
    CORE_FEATURES, BIS_STANDARDS, LITE_FEATURES, STATE_CODES
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaIntel Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stMetric { background: white; border-radius: 12px; padding: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
    .block-container { padding-top: 1rem; }
    h1 { color: #1f4e79; }
    h2, h3 { color: #2e86ab; }
    .safe-badge   { background:#27ae60; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; }
    .unsafe-badge { background:#e74c3c; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; }
    .warn-badge   { background:#f39c12; color:white; padding:3px 10px; border-radius:12px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

QUAL_COLORS = {
    "Excellent": "#27ae60",
    "Good":      "#f1c40f",
    "Poor":      "#e67e22",
    "Very Poor": "#e74c3c",
}

# ─── Cached data load ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading water quality data…")
def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    try:
        raw = load_all_csvs(data_dir)
        source = "real"
    except (FileNotFoundError, ValueError) as e:
        st.warning(f"⚠️ Could not load real data: {e}. Using synthetic demo data.")
        raw = generate_synthetic_cwc(n=8000)
        source = "synthetic"
    df = preprocess(raw)
    return df, source

@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    models = {}
    for fname in ["rf_full.pkl", "rf_lite.pkl", "xgb_full.pkl"]:
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            models[fname.replace(".pkl", "")] = joblib.load(path)
    return models

df, data_source = load_data()
models = load_models()

if data_source == "synthetic":
    st.sidebar.warning("⚠️ Using **synthetic demo data**. Upload real CSVs to `/data/` and restart.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/water.png", width=64)
st.sidebar.title("AquaIntel Analytics")
st.sidebar.markdown("*Intelligence-First Water Diagnostics*")
st.sidebar.markdown("---")

st.sidebar.header("🔍 Filters")

# State filter
states = sorted(df["state"].dropna().unique().tolist())
sel_states = st.sidebar.multiselect("State(s)", states, default=states)

# Year filter
if "year" in df.columns:
    yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
    sel_years = st.sidebar.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))
else:
    sel_years = None

# Month filter
if "month" in df.columns:
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    month_opts = list(month_map.keys())
    sel_months = st.sidebar.multiselect(
        "Month(s)", options=month_opts,
        format_func=lambda m: month_map[m],
        default=month_opts)
else:
    sel_months = None

# River filter
if "river_name" in df.columns:
    rivers = sorted(df["river_name"].dropna().unique().tolist())
    sel_rivers = st.sidebar.multiselect("River(s)", rivers, default=rivers)
else:
    sel_rivers = None

# WQI range
wqi_range = st.sidebar.slider("WQI Range", 0.0, 100.0, (0.0, 100.0))

st.sidebar.markdown("---")

# ─── Apply filters ────────────────────────────────────────────────────────────
filt = df.copy()
if sel_states:
    filt = filt[filt["state"].isin(sel_states)]
if sel_years and "year" in filt.columns:
    filt = filt[(filt["year"] >= sel_years[0]) & (filt["year"] <= sel_years[1])]
if sel_months and "month" in filt.columns:
    filt = filt[filt["month"].isin(sel_months)]
if sel_rivers and "river_name" in filt.columns:
    filt = filt[filt["river_name"].isin(sel_rivers)]
filt = filt[(filt["WQI"] >= wqi_range[0]) & (filt["WQI"] <= wqi_range[1])]

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("💧 AquaIntel Analytics")
st.markdown(f"**{len(filt):,}** records · {len(sel_states)} state(s) · "
            f"{data_source.upper()} data")
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Risk Heatmap",
    "Trends",
    "Parameter Analysis",
    "Predict",
    "Upload Data"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Records", f"{len(filt):,}")
    with col2:
        st.metric("States Covered", filt["state"].nunique())
    with col3:
        avg_wqi = filt["WQI"].mean()
        st.metric("Mean WQI", f"{avg_wqi:.1f}", delta=f"{avg_wqi-50:.1f} vs safe threshold")
    with col4:
        pct_safe = (filt["is_safe"] == 1).mean() * 100
        st.metric("% Safe Samples", f"{pct_safe:.1f}%")
    with col5:
        if "year" in filt.columns:
            st.metric("Year Span", f"{int(filt['year'].min())}–{int(filt['year'].max())}")

    st.markdown("### Water Quality Distribution")
    c1, c2 = st.columns(2)

    with c1:
        wq_counts = filt["water_quality"].value_counts().reset_index()
        wq_counts.columns = ["Category", "Count"]
        fig = px.pie(wq_counts, names="Category", values="Count",
                     color="Category", color_discrete_map=QUAL_COLORS,
                     hole=0.4, title="WQI Category Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(filt, x="WQI", nbins=40, color_discrete_sequence=["#2e86ab"],
                           title="WQI Score Distribution")
        fig.add_vline(x=50, line_dash="dash", line_color="red",
                      annotation_text="Safe threshold (50)")
        fig.update_layout(xaxis_title="WQI", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### State-wise Comparison")
    state_summary = filt.groupby("state").agg(
        mean_WQI=("WQI", "mean"),
        pct_safe=("is_safe", lambda x: x.mean() * 100),
        n=("WQI", "count")
    ).reset_index().sort_values("mean_WQI", ascending=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(state_summary, x="mean_WQI", y="state", orientation="h",
                     color="mean_WQI", color_continuous_scale="RdYlGn_r",
                     text="mean_WQI", title="Mean WQI by State (lower = better)")
        fig.update_traces(texttemplate="%{text:.1f}", textposition="inside")
        fig.add_vline(x=50, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.bar(state_summary, x="pct_safe", y="state", orientation="h",
                     color="pct_safe", color_continuous_scale="RdYlGn",
                     text="pct_safe", title="% Safe Samples by State")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
        st.plotly_chart(fig, use_container_width=True)

    # BIS Exceedance
    st.markdown("### BIS 10500 Standard Exceedance")
    exc = {}
    for param, std in BIS_STANDARDS.items():
        if param not in filt.columns:
            continue
        col = filt[param].dropna()
        if "max" in std:
            pct = (col > std["max"]).mean() * 100
        elif "min" in std:
            pct = (col < std["min"]).mean() * 100
        else:
            lo, hi = std["min"], std["max"]
            pct = ((col < lo) | (col > hi)).mean() * 100
        exc[param] = round(pct, 1)

    exc_df = pd.DataFrame(list(exc.items()), columns=["Parameter", "Exceedance %"])
    exc_df = exc_df.sort_values("Exceedance %", ascending=True)
    fig = px.bar(exc_df, x="Exceedance %", y="Parameter", orientation="h",
                 color="Exceedance %", color_continuous_scale="RdYlGn_r",
                 title="% Samples Exceeding BIS Drinking Water Standard")
    fig.add_vline(x=20, line_dash="dot", line_color="orange",
                  annotation_text="20% alert level")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK HEATMAP
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🗺️ Spatial Risk Heatmap")

    if "latitude" not in filt.columns or "longitude" not in filt.columns:
        st.warning("No latitude/longitude columns in dataset for mapping.")
    else:
        map_df = filt.dropna(subset=["latitude", "longitude", "WQI"]).copy()
        map_df = map_df[(map_df["latitude"].between(5, 38)) &
                        (map_df["longitude"].between(67, 98))]

        col_param = st.selectbox("Color by parameter",
                                  ["WQI"] + [f for f in CORE_FEATURES if f in filt.columns])

        if len(map_df) > 0:
            fig = px.density_mapbox(
                map_df, lat="latitude", lon="longitude",
                z=col_param, radius=20,
                center={"lat": 20.5, "lon": 80}, zoom=4,
                mapbox_style="carto-positron",
                color_continuous_scale="RdYlGn_r",
                title=f"Risk Heatmap — {col_param}",
                hover_data={k: True for k in
                            ["state", "river_name", "station_name", "WQI",
                             "water_quality", "year"]
                            if k in map_df.columns})
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            # Scatter map below
            st.markdown("#### Station-level Scatter Map")
            fig2 = px.scatter_mapbox(
                map_df.sample(min(2000, len(map_df)), random_state=42),
                lat="latitude", lon="longitude",
                color="water_quality",
                color_discrete_map=QUAL_COLORS,
                size="WQI", size_max=12,
                mapbox_style="carto-positron",
                center={"lat": 20.5, "lon": 80}, zoom=4,
                hover_data={k: True for k in
                            ["state", "river_name", "station_name", "WQI", "year"]
                            if k in map_df.columns},
                title="Water Quality Stations")
            fig2.update_layout(height=550)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No valid coordinates after filtering.")

    # State risk summary
    st.markdown("### State-level Risk Summary")
    risk_tbl = filt.groupby("state").agg(
        Records=("WQI", "count"),
        Mean_WQI=("WQI", "mean"),
        Pct_VeryPoor=("water_quality",
                      lambda x: (x.astype(str) == "Very Poor").mean() * 100),
        Pct_Safe=("is_safe", lambda x: x.mean() * 100),
    ).round(2).reset_index()
    risk_tbl["Risk Level"] = risk_tbl["Mean_WQI"].apply(
        lambda v: "🔴 High" if v > 70 else "🟡 Medium" if v > 50 else "🟢 Low")
    st.dataframe(risk_tbl.sort_values("Mean_WQI", ascending=False),
                 use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRENDS & SEASONAL
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Temporal Trends")

    param_trend = st.selectbox("Parameter",
                                ["WQI"] + [f for f in CORE_FEATURES if f in filt.columns])

    if "year" in filt.columns:
        trend = filt.groupby(["year", "state"])[param_trend].median().reset_index()
        fig = px.line(trend, x="year", y=param_trend, color="state",
                      markers=True, title=f"Yearly Trend — {param_trend} by State")
        if param_trend in BIS_STANDARDS:
            std = BIS_STANDARDS[param_trend]
            if "max" in std:
                fig.add_hline(y=std["max"], line_dash="dash", line_color="red",
                              annotation_text=f"BIS max {std['max']}")
            elif "min" in std:
                fig.add_hline(y=std["min"], line_dash="dash", line_color="green",
                              annotation_text=f"BIS min {std['min']}")
        st.plotly_chart(fig, use_container_width=True)

    if "month" in filt.columns:
        st.markdown("### 🌧️ Seasonal Patterns")
        month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        seasonal = filt.groupby("month")[param_trend].agg(["median","mean","std"]).reset_index()
        seasonal["month_name"] = seasonal["month"].map(month_map)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=seasonal["month_name"], y=seasonal["median"],
            mode="lines+markers", name="Median",
            line=dict(color="#1f4e79", width=2),
            marker=dict(size=8)))
        fig.add_trace(go.Scatter(
            x=list(seasonal["month_name"]) + list(seasonal["month_name"])[::-1],
            y=list(seasonal["median"] + seasonal["std"]) +
              list((seasonal["median"] - seasonal["std"]).clip(0))[::-1],
            fill="toself", fillcolor="rgba(30,130,200,0.15)",
            line=dict(color="rgba(255,255,255,0)"), name="±1 SD"))
        if param_trend in BIS_STANDARDS:
            std_val = BIS_STANDARDS[param_trend]
            if "max" in std_val:
                fig.add_hline(y=std_val["max"], line_dash="dash", line_color="red",
                              annotation_text=f"BIS max={std_val['max']}")
        fig.update_layout(title=f"Seasonal Pattern — {param_trend}",
                          xaxis_title="Month", yaxis_title=param_trend)
        st.plotly_chart(fig, use_container_width=True)

        # Monsoon vs non-monsoon comparison
        st.markdown("### ⛈️ Monsoon vs Non-Monsoon")
        filt["season"] = filt["month"].apply(
            lambda m: "Monsoon (Jun–Sep)" if m in [6,7,8,9] else "Non-Monsoon")
        show_params = [f for f in ["pH","dissolved_oxygen","BOD","turbidity",
                                    "nitrates","TDS","conductivity"] if f in filt.columns]
        melt = filt[show_params + ["season"]].melt(id_vars="season",
                                                    var_name="Parameter", value_name="Value")
        fig = px.box(melt, x="Parameter", y="Value", color="season",
                     color_discrete_map={"Monsoon (Jun–Sep)":"#2e86ab","Non-Monsoon":"#f18f01"},
                     title="Monsoon vs Non-Monsoon Parameter Comparison")
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — PARAMETER ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🔬 Deep Parameter Analysis")

    avail = [f for f in CORE_FEATURES if f in filt.columns]
    param_a = st.selectbox("Parameter A", avail, index=0)
    param_b = st.selectbox("Parameter B",
                           [f for f in avail if f != param_a],
                           index=min(1, len(avail)-2))

    c1, c2 = st.columns(2)
    with c1:
        # Distribution
        fig = px.histogram(filt, x=param_a, color="water_quality",
                           color_discrete_map=QUAL_COLORS, nbins=30,
                           barmode="overlay", opacity=0.7,
                           title=f"Distribution of {param_a} by Quality")
        if param_a in BIS_STANDARDS:
            std = BIS_STANDARDS[param_a]
            if "max" in std:
                fig.add_vline(x=std["max"], line_dash="dash", line_color="red",
                              annotation_text=f"BIS max={std['max']}")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Scatter A vs B coloured by WQI
        scatter_df = filt[[param_a, param_b, "WQI", "state", "water_quality"]].dropna()
        fig = px.scatter(scatter_df.sample(min(2000, len(scatter_df)), random_state=42),
                         x=param_a, y=param_b,
                         color="water_quality", color_discrete_map=QUAL_COLORS,
                         hover_data=["state", "WQI"],
                         opacity=0.6, title=f"{param_a} vs {param_b}")
        if param_a in BIS_STANDARDS and "max" in BIS_STANDARDS[param_a]:
            fig.add_vline(x=BIS_STANDARDS[param_a]["max"], line_dash="dash", line_color="red")
        if param_b in BIS_STANDARDS and "max" in BIS_STANDARDS[param_b]:
            fig.add_hline(y=BIS_STANDARDS[param_b]["max"], line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation matrix
    st.markdown("### Correlation Heatmap")
    show_corr = st.multiselect("Select parameters for correlation",
                               options=avail, default=avail[:min(10, len(avail))])
    if len(show_corr) >= 2:
        corr = filt[show_corr + ["WQI"]].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, title="Pearson Correlation Matrix",
                        aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    # Stats table
    st.markdown("### Summary Statistics")
    stats_cols = [f for f in avail if f in filt.columns] + ["WQI"]
    stats_df = filt[stats_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T.round(3)
    stats_df["% missing"] = (filt[stats_cols].isna().mean() * 100).round(1)
    bis_col = []
    for c in stats_df.index:
        s = BIS_STANDARDS.get(c, {})
        val = s.get("max", s.get("min", "—"))
        bis_col.append(val)
    stats_df["BIS Limit"] = bis_col
    st.dataframe(stats_df, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🤖 Water Quality Prediction")
    st.markdown("Enter water sample values to get an instant RFWQI safety assessment.")

    model_choice = st.radio("Model", ["RF Lite (3 parameters)", "RF Full", "XGB Full"],
                            horizontal=True)

    if model_choice == "RF Lite (3 parameters)":
        model_key = "rf_lite"
        feature_list = [f for f in LITE_FEATURES if f in df.columns]
        st.info("⚡ Lite model — uses only pH, Conductivity, Nitrates. "
                "Ideal for low-cost / rural settings.")
    elif model_choice == "XGB Full":
        model_key = "xgb_full"
        feature_list = [f for f in CORE_FEATURES if f in df.columns]
    else:
        model_key = "rf_full"
        feature_list = [f for f in CORE_FEATURES if f in df.columns]

    if not models:
        st.warning("⚠️ No trained models found. Run `python notebooks/model_dev.py` first. "
                   "Showing manual WQI calculator instead.")
        # Fallback: manual WQI calculator
        st.markdown("#### Manual WQI Calculator")
    elif model_key not in models:
        st.warning(f"Model `{model_key}` not found. Run model_dev.py to train it.")
    else:
        st.markdown("#### Enter Sample Parameters")
        model_obj  = models[model_key]["model"]
        feat_names = models[model_key]["features"]

        # Group features for better UI
        input_vals = {}
        cols = st.columns(3)
        for i, feat in enumerate(feat_names):
            col = cols[i % 3]
            # Smart defaults from dataset median
            default = float(df[feat].median()) if feat in df.columns else 7.0
            bis_hint = ""
            if feat in BIS_STANDARDS:
                s = BIS_STANDARDS[feat]
                bis_hint = (f" (BIS max: {s['max']})" if "max" in s else
                            f" (BIS min: {s['min']})" if "min" in s else "")
            val = col.number_input(f"{feat}{bis_hint}", value=round(default, 3),
                                   format="%.3f", key=feat)
            input_vals[feat] = val

        if st.button("🔬 Analyse Sample", type="primary"):
            input_df = pd.DataFrame([input_vals])
            pred     = model_obj.predict(input_df)[0]
            prob     = model_obj.predict_proba(input_df)[0]

            safe_prob   = prob[1] * 100
            unsafe_prob = prob[0] * 100

            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                label = "✅ SAFE" if pred == 1 else "❌ UNSAFE"
                badge = "safe-badge" if pred == 1 else "unsafe-badge"
                st.markdown(f"### Prediction\n<span class='{badge}'>{label}</span>",
                            unsafe_allow_html=True)
            with c2:
                st.metric("Safe Probability",   f"{safe_prob:.1f}%")
            with c3:
                st.metric("Unsafe Probability", f"{unsafe_prob:.1f}%")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=safe_prob,
                title={"text": "Safety Score (%)"},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#2e86ab"},
                    "steps": [
                        {"range": [0, 30],  "color": "#e74c3c"},
                        {"range": [30, 60], "color": "#f39c12"},
                        {"range": [60, 100],"color": "#27ae60"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3},
                                  "thickness": 0.75, "value": 50},
                }
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

            # Parameter compliance check
            st.markdown("### BIS Compliance Check")
            compliance = []
            for feat, val in input_vals.items():
                std = BIS_STANDARDS.get(feat, {})
                if not std:
                    continue
                if "max" in std and val > std["max"]:
                    status = "❌ Exceeds limit"
                    severity = "High" if val > std["max"] * 2 else "Moderate"
                elif "min" in std and val < std["min"]:
                    status = "❌ Below limit"
                    severity = "High"
                else:
                    status = "✅ Within limit"
                    severity = "—"
                compliance.append({
                    "Parameter": feat,
                    "Your Value": round(val, 3),
                    "BIS Standard": std.get("max", std.get("min", "—")),
                    "Status": status,
                    "Severity": severity,
                })
            if compliance:
                comp_df = pd.DataFrame(compliance)
                st.dataframe(comp_df, use_container_width=True)

    # WQI calculator (always shown)
    st.markdown("---")
    st.markdown("###  Manual WQI Calculator")
    with st.expander("Compute WQI from raw values", expanded=False):
        calc_vals = {}
        cols = st.columns(4)
        for i, (param, std) in enumerate(BIS_STANDARDS.items()):
            if param not in df.columns:
                continue
            default = float(df[param].median()) if param in df.columns else 0.0
            hint = std.get("max", std.get("min", ""))
            val = cols[i % 4].number_input(
                f"{param} (BIS: {hint})", value=round(default, 3),
                format="%.3f", key=f"calc_{param}")
            calc_vals[param] = val

        if st.button("Calculate WQI"):
            from utils.data_loader import compute_wqi, label_water_quality
            tmp = pd.DataFrame([calc_vals])
            wqi_val = compute_wqi(tmp).iloc[0]
            cat = label_water_quality(pd.Series([wqi_val])).iloc[0]
            st.metric("WQI Score", f"{wqi_val:.2f}",
                      delta=f"{wqi_val-50:.2f} vs safe threshold (50)")
            badge_map = {"Excellent":"safe-badge","Good":"safe-badge",
                         "Poor":"warn-badge","Very Poor":"unsafe-badge"}
            badge = badge_map.get(str(cat), "warn-badge")
            st.markdown(f"Category: <span class='{badge}'>{cat}</span>",
                        unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — UPLOAD NEW DATA
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 📤 Upload New Dataset")
    st.markdown(
        "Upload any CWC-format CSV. Columns are auto-detected and normalised. "
        "Multiple files can be saved to `/data/` for permanent inclusion."
    )

    uploaded = st.file_uploader("Upload CSV file(s)", type=["csv"],
                                 accept_multiple_files=True)
    if uploaded:
        frames = []
        for f in uploaded:
            try:
                tmp = pd.read_csv(f, low_memory=False)
                frames.append(tmp)
                st.success(f"✅ {f.name}: {tmp.shape[0]:,} rows, {tmp.shape[1]} cols")
            except Exception as e:
                st.error(f"❌ {f.name}: {e}")

        if frames:
            import io
            from utils.data_loader import _normalise_columns, preprocess as pp
            new_df = pd.concat(frames, ignore_index=True)
            new_df = _normalise_columns(new_df)
            new_df = pp(new_df)

            st.markdown(f"**Combined: {new_df.shape[0]:,} rows, {new_df.shape[1]} cols**")
            st.dataframe(new_df.head(50), use_container_width=True)

            # Quick stats
            st.markdown("#### Quick Stats")
            avail = [f for f in CORE_FEATURES if f in new_df.columns]
            if avail:
                st.dataframe(new_df[avail + ["WQI"]].describe().round(3),
                             use_container_width=True)

            # WQI distribution
            fig = px.histogram(new_df, x="WQI", nbins=30,
                               color_discrete_sequence=["#2e86ab"],
                               title="WQI Distribution — Uploaded Data")
            fig.add_vline(x=50, line_dash="dash", line_color="red",
                          annotation_text="Safe threshold")
            st.plotly_chart(fig, use_container_width=True)

            # Save option
            save_name = st.text_input("Save filename (optional)",
                                      value="uploaded_dataset.csv")
            if st.button("💾 Save to /data/"):
                save_path = os.path.join(os.path.dirname(__file__),
                                         "data", save_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                new_df.to_csv(save_path, index=False)
                st.success(f"Saved to {save_path}. Restart the app to include it.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")

