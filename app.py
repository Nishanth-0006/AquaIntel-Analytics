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
from huggingface_hub import hf_hub_download
import joblib

from utils.data_loader import (
    load_all_csvs, generate_synthetic_cwc, preprocess,
    CORE_FEATURES, BIS_STANDARDS, RENAME_MAP,
)
from utils.model_utils import SoftVotingHybrid

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaIntel Analytics — CWC Water Quality",
    page_icon="💧",
    layout="wide",
)

# ─── SECTION: Theme State Init ──────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ─── SECTION: Global CSS (Professional Navy / Water Theme) ──────────────────────
def inject_theme_css(dark: bool):
    # ── LIGHT: clean white + deep navy accents (presentation-grade) ──
    # ── DARK:  deep ocean navy palette ──
    if dark:
        bg         = "#060d1a"
        bg2        = "#0d1f38"
        card_bg    = "#0d1f38"
        card_bg2   = "#112240"
        border     = "#1e3a5f"
        text       = "#e2eaf5"
        text2      = "#8aadcc"
        text3      = "#5a7fa0"
        tab_bg     = "#0d1f38"
        tab_border = "#1e3a5f"
        tab_sel    = "#38bdf8"
        input_bg   = "rgba(255,255,255,0.05)"
        input_bord = "#1e3a5f"
        exp_bg     = "#0d1f38"
        metric_val = "#e2eaf5"
        metric_lbl = "#8aadcc"
        card_shad  = "rgba(0,0,0,0.5)"
        badge_row  = "#0a1628"
    else:
        bg         = "#f0f4f9"
        bg2        = "#ffffff"
        card_bg    = "#ffffff"
        card_bg2   = "#f7fafd"
        border     = "#d6e3f0"
        text       = "#0a1f3d"
        text2      = "#2c4a6e"
        text3      = "#5a7fa0"
        tab_bg     = "#ffffff"
        tab_border = "#d6e3f0"
        tab_sel    = "#1e3a8a"
        input_bg   = "#ffffff"
        input_bord = "#b8d0e8"
        exp_bg     = "#f7fafd"
        metric_val = "#0a1f3d"
        metric_lbl = "#2c4a6e"
        card_shad  = "rgba(10,31,61,0.08)"
        badge_row  = "#eef4fb"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ══ BASE RESET ══════════════════════════════════════════════════════════════ */
html, body, [class*="css"], .stApp {{
    font-family: 'Inter', -apple-system, sans-serif !important;
    background-color: {bg} !important;
    color: {text} !important;
    font-size: 14px !important;
    -webkit-font-smoothing: antialiased !important;
}}
.stApp > header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {{
    display: none !important;
    height: 0 !important;
}}
.block-container {{
    padding-top: 0.25rem !important;
    padding-left: 1.1rem !important;
    padding-right: 1.1rem !important;
    padding-bottom: 2rem !important;
    background-color: {bg} !important;
    max-width: 100% !important;
}}

/* ══ SIDEBAR ═════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #061A2E 0%, #0A2540 52%, #075985 100%) !important;
    border-right: 1px solid rgba(202,240,248,0.18) !important;
    margin-top: 0 !important;
}}
[data-testid="stSidebar"] > div {{ padding: 0 !important; }}
[data-testid="stSidebar"] * {{ color: #EAFBFF !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b {{ color: #ffffff !important; }}

/* Sidebar section labels */
[data-testid="stSidebar"] p {{
    color: #D6F3FF !important;
    font-size: 0.78rem !important;
    line-height: 1.5 !important;
}}

/* Sidebar inputs */
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    background-color: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(202,240,248,0.36) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] [data-baseweb="input"] > div {{
    background-color: rgba(255,255,255,0.96) !important;
    border: 1px solid rgba(202,240,248,0.75) !important;
    color: #061A2E !important;
    border-radius: 8px !important;
}}
[data-testid="stSidebar"] input {{
    color: #061A2E !important;
    -webkit-text-fill-color: #061A2E !important;
    background: transparent !important;
}}
[data-testid="stSidebar"] input::placeholder {{ color: #31546A !important; }}

/* Sidebar multiselect tags */
[data-testid="stSidebar"] span[data-baseweb="tag"] {{
    background: rgba(202,240,248,0.18) !important;
    color: #ffffff !important;
    border: 1px solid rgba(202,240,248,0.38) !important;
    border-radius: 20px !important;
    font-size: 0.75rem !important;
}}

/* Sidebar slider track */
[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {{
    background-color: #38bdf8 !important;
}}

/* Sidebar expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(202,240,248,0.28) !important;
    border-radius: 10px !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p {{
    color: #EAFBFF !important;
    font-weight: 600 !important;
}}

/* Sidebar info box */
[data-testid="stSidebar"] [data-testid="stInfo"] {{
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(202,240,248,0.28) !important;
    border-radius: 8px !important;
    color: #7dd3fc !important;
}}
[data-testid="stSidebar"] [data-testid="stInfo"] p {{
    color: #D6F3FF !important;
    font-size: 0.78rem !important;
}}

/* Sidebar buttons — base */
[data-testid="stSidebar"] .stButton > button {{
    background: #ffffff !important;
    color: #061A2E !important;
    border: 1px solid rgba(202,240,248,0.36) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}}
[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button div {{
    color: #061A2E !important;
    -webkit-text-fill-color: #061A2E !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: #F1FAFF !important;
    color: #061A2E !important;
}}

/* Sidebar warning */
[data-testid="stSidebar"] [data-testid="stWarning"] {{
    background: rgba(245,158,11,0.1) !important;
    border-left: 3px solid #f59e0b !important;
    border-radius: 6px !important;
}}
[data-testid="stSidebar"] [data-testid="stWarning"] p {{
    color: #fcd34d !important;
    font-size: 0.76rem !important;
}}

/* ══ MAIN CONTENT TEXT ════════════════════════════════════════════════════════ */
h1 {{ color: {text} !important; font-weight: 800 !important; letter-spacing: -0.03em !important; }}
h2 {{ color: {text} !important; font-weight: 700 !important; }}
h3 {{ color: {text} !important; font-weight: 600 !important; }}
h4, h5, h6 {{ color: {text} !important; }}
p, .stMarkdown p {{ color: {text2} !important; line-height: 1.7 !important; }}
label, .stTextInput label, .stNumberInput label,
.stSelectbox label, .stSlider label,
.stMultiSelect label {{ color: {text2} !important; font-weight: 500 !important; font-size: 0.82rem !important; }}
.stCaption, small {{ color: {text3} !important; font-size: 0.8rem !important; }}
hr {{ border-color: {border} !important; opacity: 0.6 !important; }}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
    color: #D6F3FF !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="select"] svg {{
    color: #ffffff !important;
    fill: #ffffff !important;
}}
[data-testid="stSidebar"] [data-testid="stSlider"] * {{
    color: #EAFBFF !important;
}}

/* ══ INPUT FIELDS ════════════════════════════════════════════════════════════ */
[data-baseweb="input"] > div,
.stNumberInput input,
.stTextInput input {{
    background-color: {input_bg} !important;
    border: 1px solid {input_bord} !important;
    border-radius: 8px !important;
    color: {text} !important;
    font-size: 0.88rem !important;
}}
[data-baseweb="select"] > div {{
    background-color: {input_bg} !important;
    border: 1px solid {input_bord} !important;
    border-radius: 8px !important;
    color: {text} !important;
}}
input::placeholder {{ color: {text3} !important; }}

/* ══ TABS ════════════════════════════════════════════════════════════════════ */
div[role="tablist"] {{
    background: {tab_bg} !important;
    border-bottom: 2px solid {tab_border} !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 0 0.5rem !important;
    gap: 0.2rem !important;
}}
button[role="tab"] {{
    background: transparent !important;
    border: none !important;
    border-radius: 6px 6px 0 0 !important;
    color: {text3} !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 0.7rem 1.1rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.15s !important;
}}
button[role="tab"][aria-selected="true"] {{
    color: {tab_sel} !important;
    font-weight: 700 !important;
    border-bottom: 3px solid {tab_sel} !important;
    background: {'rgba(56,189,248,0.06)' if dark else 'rgba(30,58,138,0.05)'} !important;
}}
button[role="tab"]:hover {{ color: {tab_sel} !important; }}

/* ══ BUTTONS ═════════════════════════════════════════════════════════════════ */
.stButton > button {{
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 0.55rem 1.6rem !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 2px 8px rgba(30,58,138,0.3) !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(30,58,138,0.4) !important;
}}
.stDownloadButton > button,
[data-testid="stDownloadButton"] > button {{
    background: linear-gradient(135deg, #15803d 0%, #16a34a 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(22,163,74,0.3) !important;
}}

/* ══ CARDS / BORDERED CONTAINERS ════════════════════════════════════════════ */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px {card_shad} !important;
    padding: 0.2rem !important;
}}

/* ══ METRICS ═════════════════════════════════════════════════════════════════ */
[data-testid="stMetricValue"] {{ color: {metric_val} !important; font-weight: 800 !important; }}
[data-testid="stMetricLabel"] {{ color: {metric_lbl} !important; font-weight: 500 !important; }}
[data-testid="metric-container"] {{
    background: {card_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}}

/* ══ DATAFRAMES ══════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {{ border-radius: 10px !important; overflow: hidden !important; }}
[data-testid="stDataFrame"] div {{ color: {text} !important; }}
[data-testid="stDataFrame"] th {{
    background: {tab_bg} !important;
    color: {text} !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    border-bottom: 2px solid {border} !important;
}}

/* ══ EXPANDERS ═══════════════════════════════════════════════════════════════ */
.streamlit-expanderHeader {{
    background-color: {exp_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 8px !important;
    color: {text2} !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
}}
.streamlit-expanderContent {{
    background-color: {card_bg} !important;
    border: 1px solid {border} !important;
    border-top: none !important;
    color: {text} !important;
}}

/* ══ FILE UPLOADER ═══════════════════════════════════════════════════════════ */
[data-testid="stFileUploadDropzone"] {{
    background: {'rgba(56,189,248,0.04)' if dark else '#f0f7ff'} !important;
    border: 2px dashed {'rgba(56,189,248,0.3)' if dark else '#93c5fd'} !important;
    border-radius: 12px !important;
    color: {text2} !important;
}}

/* ══ ALERTS ══════════════════════════════════════════════════════════════════ */
.stAlert {{ border-radius: 10px !important; }}
[data-testid="stSuccess"] {{
    background: {'rgba(22,163,74,0.12)' if dark else '#f0fdf4'} !important;
    border: 1px solid {'rgba(22,163,74,0.3)' if dark else '#86efac'} !important;
    border-radius: 10px !important;
    color: {'#86efac' if dark else '#15803d'} !important;
}}
[data-testid="stError"] {{
    background: {'rgba(220,38,38,0.12)' if dark else '#fef2f2'} !important;
    border: 1px solid {'rgba(220,38,38,0.3)' if dark else '#fca5a5'} !important;
    border-radius: 10px !important;
    color: {'#fca5a5' if dark else '#b91c1c'} !important;
}}
[data-testid="stWarning"] {{
    background: {'rgba(245,158,11,0.12)' if dark else '#fffbeb'} !important;
    border: 1px solid {'rgba(245,158,11,0.3)' if dark else '#fde68a'} !important;
    border-radius: 10px !important;
}}
[data-testid="stInfo"] {{
    background: {'rgba(56,189,248,0.1)' if dark else '#eff6ff'} !important;
    border: 1px solid {'rgba(56,189,248,0.25)' if dark else '#bfdbfe'} !important;
    border-radius: 10px !important;
}}

/* ══ DROPDOWN MENUS ══════════════════════════════════════════════════════════ */
[data-baseweb="popover"],
[data-baseweb="menu"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 10px !important;
    box-shadow: 0 8px 24px {card_shad} !important;
}}
[data-baseweb="option"] {{
    background-color: {card_bg} !important;
    color: {text} !important;
    font-size: 0.85rem !important;
}}
[data-baseweb="option"]:hover {{
    background-color: {'rgba(56,189,248,0.1)' if dark else '#eff6ff'} !important;
}}

/* ══ SPINNER ═════════════════════════════════════════════════════════════════ */
[data-testid="stSpinner"] p {{ color: {text2} !important; }}

/* Upload risk module */
.risk-hero {{
    background: linear-gradient(135deg, #0A2540 0%, #0077B6 58%, #00B4D8 100%);
    border: 1px solid rgba(202,240,248,0.25);
    border-radius: 14px;
    box-shadow: 0 12px 30px rgba(10,37,64,0.18);
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}}
.risk-hero h3 {{
    color: #CAF0F8 !important;
    margin: 0 0 0.25rem 0 !important;
}}
.risk-hero p {{
    color: rgba(255,255,255,0.86) !important;
    margin: 0 !important;
}}
.risk-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: center;
    margin: 0.4rem 0 0.8rem 0;
}}
.risk-legend span {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    color: {text2};
    font-size: 0.84rem;
    font-weight: 600;
}}
.risk-dot {{
    width: 11px;
    height: 11px;
    border-radius: 50%;
    display: inline-block;
}}
.stApp .stButton > button {{
    background: #ffffff !important;
    color: #0A2540 !important;
    border: 1px solid #b8d0e8 !important;
    box-shadow: 0 2px 8px rgba(10,37,64,0.12) !important;
}}
.stApp .stButton > button:hover {{
    background: #F1FAFF !important;
    color: #0A2540 !important;
    border-color: #0077B6 !important;
}}
</style>
""", unsafe_allow_html=True)

inject_theme_css(st.session_state.dark_mode)

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
    """Apply consistent theme-aware layout to every Plotly chart."""
    dark = st.session_state.get("dark_mode", False)
    paper = "#1e293b" if dark else "#ffffff"
    plot  = "#0f172a" if dark else "#f8fafc"
    grid  = "#334155" if dark else "#e2e8f0"
    txt   = "#f1f5f9" if dark else "#0f172a"
    txt2  = "#94a3b8" if dark else "#475569"
    fig.update_layout(
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        font=dict(color=txt, family="Inter, sans-serif", size=13),
        title=dict(font=dict(color=txt, size=16), x=0, xanchor="left"),
        legend=dict(font=dict(color=txt2, size=12), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(
            gridcolor=grid, zerolinecolor=grid,
            tickfont=dict(color=txt2, size=12),
            title_font=dict(color=txt, size=13),
        ),
        yaxis=dict(
            gridcolor=grid, zerolinecolor=grid,
            tickfont=dict(color=txt2, size=12),
            title_font=dict(color=txt, size=13),
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
    """Haversine-based nearest monitoring station finder (unique locations)."""

    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    df2 = df.dropna(subset=["latitude", "longitude"]).copy()

    # 🔥 IMPORTANT FIX: remove duplicate locations
    df2 = df2.drop_duplicates(subset=["latitude", "longitude"])

    df2["distance_km"] = df2.apply(
        lambda r: haversine(lat, lon, r["latitude"], r["longitude"]),
        axis=1
    )

    return df2.nsmallest(n, "distance_km")


def section_header(text):
    """Render an uppercase section divider label."""
    dark = st.session_state.get("dark_mode", False)
    col  = "#64748b" if dark else "#94a3b8"
    st.markdown(
        f'<div style="font-size:0.72rem;font-weight:700;color:{col};'
        f'text-transform:uppercase;letter-spacing:0.1em;margin:1.4rem 0 0.6rem 0;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label, value, sub, color="#0f172a", badge=None,
             badge_color="#16a34a", badge_bg="#dcfce7", delta_text="",
             delta_color="#94a3b8"):
    dark = st.session_state.get("dark_mode", False)
    card_bg  = "#1e293b" if dark else "#ffffff"
    card_brd = "#334155" if dark else "#e2e8f0"
    lbl_col  = "#94a3b8" if dark else "#64748b"
    sub_col  = "#94a3b8" if dark else "#64748b"
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
<div style="background:{card_bg};border:1px solid {card_brd};border-radius:12px;
            padding:1.2rem 1.4rem;box-shadow:0 2px 8px rgba(0,0,0,0.06);height:100%;">
  <div style="font-size:0.72rem;font-weight:600;color:{lbl_col};text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:0.6rem;">{label}</div>
  <div style="font-size:2.4rem;font-weight:800;color:{color};line-height:1.1;">{value}</div>
  <div style="font-size:0.82rem;color:{sub_col};margin-top:0.4rem;">{sub}{badge_html}</div>
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
    from huggingface_hub import hf_hub_download
    import json

    try:
        path = hf_hub_download(
            repo_id="crushh3/aquaintel-data",
            filename="data/geo/india_districts.geojson",
            repo_type="dataset"
        )

        with open(path, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        # 🔥 CRITICAL FIX: add plotly_key to every feature
        for feature in geojson["features"]:
            props = feature.get("properties", {})

            district = first_property(
                props,
                ["district", "DISTRICT", "district_name", "NAME_2", "name"]
            )

            props["plotly_key"] = str(district).lower().strip()

        return geojson

    except Exception as e:
        st.error(f"GeoJSON failed: {e}")
        return None


@st.cache_data
def build_district_choropleth_frame(frame):
    if frame.empty or "WQI" not in frame.columns:
        return pd.DataFrame(columns=["state", "district", "WQI", "plotly_key"])

    district_col = next((c for c in ["district","District","district_name"] if c in frame.columns), None)
    state_col    = next((c for c in ["state","State","state_name"] if c in frame.columns), None)

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

    # 🔥 CRITICAL FIX: normalize names
    district_df["district"] = (
        district_df["district"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    district_df["state"] = (
        district_df["state"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # aggregate
    district_df = (
        district_df
        .groupby(["state","district"], as_index=False)["WQI"]
        .mean()
        .sort_values("WQI", ascending=False)
    )

    district_df["plotly_key"] = (
        district_df["district"]
        .astype(str)
        .str.lower()
        .str.strip()
    )
    return district_df

UPLOAD_COLUMN_CANDIDATES = {
    "district": [
        "district", "districtname", "district_name", "location", "locationname",
        "station", "stationname", "monitoringstation", "samplinglocation", "place",
    ],
    "pH": ["ph", "potentialofhydrogenph", "potentialofhydrogen", "phvalue", "phlevel"],
    "conductivity": [
        "conductivity", "electricalconductivity", "electricalconductivityuscm",
        "electricalconductivitymicrosiemenscm", "ec", "ecuscm", "specificconductance",
        "specificconductivity",
    ],
    "nitrate": [
        "nitrate", "nitratemgl", "nitrates", "no3", "no3n", "nitraten",
        "nitratenmgnl", "nitritennitraten", "nitritennitratenmgnl",
    ],
    "latitude": ["latitude", "lat", "y", "gpslatitude"],
    "longitude": ["longitude", "lon", "long", "lng", "x", "gpslongitude"],
}

DISTRICT_COORDINATES = {
    "anantapur": (14.6819, 77.6006), "anantapuramu": (14.6819, 77.6006),
    "annamayya": (14.2400, 78.7500), "chittoor": (13.2172, 79.1003),
    "cuddapah": (14.4673, 78.8242), "kadapa": (14.4673, 78.8242),
    "ysr": (14.4673, 78.8242), "east godavari": (17.0005, 81.8040),
    "eluru": (16.7107, 81.0952), "guntur": (16.3067, 80.4365),
    "kakinada": (16.9891, 82.2475), "konaseema": (16.5787, 82.0061),
    "krishna": (16.6100, 80.7214), "kurnool": (15.8281, 78.0373),
    "manyam": (18.7500, 83.4500), "nandyal": (15.4786, 78.4831),
    "nellore": (14.4426, 79.9865), "spsr nellore": (14.4426, 79.9865),
    "ntr": (16.5062, 80.6480), "palnadu": (16.2350, 79.7400),
    "prakasam": (15.3485, 79.5603), "srikakulam": (18.2949, 83.8938),
    "sri sathya sai": (14.1670, 77.8110), "tirupati": (13.6288, 79.4192),
    "visakhapatnam": (17.6868, 83.2185), "anakapalli": (17.6913, 83.0039),
    "alluri sitharama raju": (17.8500, 82.6500), "vizianagaram": (18.1067, 83.3956),
    "west godavari": (16.9174, 81.3399), "adilabad": (19.6641, 78.5320),
    "bhadradri kothagudem": (17.5544, 80.6197), "hyderabad": (17.3850, 78.4867),
    "jagtial": (18.7909, 78.9119), "jayashankar bhupalpally": (18.4294, 79.8634),
    "jogulamba gadwal": (16.2350, 77.7956), "kamareddy": (18.3205, 78.3370),
    "karimnagar": (18.4386, 79.1288), "khammam": (17.2473, 80.1514),
    "mahbubnagar": (16.7375, 78.0081), "medak": (18.0451, 78.2608),
    "nalgonda": (17.0575, 79.2684), "nizamabad": (18.6725, 78.0941),
    "rangareddy": (17.3891, 78.0302), "sangareddy": (17.6248, 78.0867),
    "siddipet": (18.1018, 78.8520), "suryapet": (17.1405, 79.6236),
    "warangal": (17.9689, 79.5941),
}


def normalize_upload_column(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def find_upload_column(columns, logical_name):
    normalized = {normalize_upload_column(c): c for c in columns}
    for candidate in UPLOAD_COLUMN_CANDIDATES[logical_name]:
        key = normalize_upload_column(candidate)
        if key in normalized:
            return normalized[key]
    for col in columns:
        key = normalize_upload_column(col)
        if any(normalize_upload_column(candidate) in key for candidate in UPLOAD_COLUMN_CANDIDATES[logical_name]):
            return col
    return None


def find_upload_parameter_columns(columns):
    alias_map = {}
    exact_rename_keys = {normalize_upload_column(raw_name) for raw_name in RENAME_MAP}
    for raw_name, canonical in RENAME_MAP.items():
        if canonical in BIS_STANDARDS:
            alias_map[normalize_upload_column(raw_name)] = canonical
    for canonical in BIS_STANDARDS:
        alias_map[normalize_upload_column(canonical)] = canonical
        alias_map[normalize_upload_column(PARAM_LABELS.get(canonical, canonical))] = canonical

    matched = {}
    for col in columns:
        key = normalize_upload_column(col)
        canonical = alias_map.get(key)
        if canonical is None and key in exact_rename_keys:
            continue
        if canonical is None:
            canonical = next(
                (param for alias, param in alias_map.items() if len(alias) >= 4 and alias in key),
                None,
            )
        if canonical and canonical not in matched:
            matched[canonical] = col
    return matched


def parameter_violation_mask(series, standard):
    values = pd.to_numeric(series, errors="coerce")
    mask = pd.Series(False, index=values.index)
    if "min" in standard:
        mask = mask | (values < standard["min"])
    if "max" in standard:
        mask = mask | (values > standard["max"])
    return mask.fillna(False)


def read_uploaded_water_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        return pd.DataFrame()
    raise ValueError("Unsupported file format. Upload a CSV, XLSX, or XLS file.")


def prepare_uploaded_risk_data(raw_df):
    if raw_df.empty:
        return pd.DataFrame(), {}, ["Uploaded file has no rows."]

    column_map = {
        key: find_upload_column(raw_df.columns, key)
        for key in ["district", "pH", "latitude", "longitude"]
    }
    missing = []
    if column_map.get("district") is None:
        missing.append("district")
    if column_map.get("pH") is None:
        missing.append("pH")
    if missing:
        return pd.DataFrame(), column_map, missing

    out = pd.DataFrame({
        "District": raw_df[column_map["district"]].astype(str).str.strip(),
        "pH": pd.to_numeric(raw_df[column_map["pH"]], errors="coerce"),
    })
    out["District"] = out["District"].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    if column_map.get("latitude") and column_map.get("longitude"):
        out["latitude"] = pd.to_numeric(raw_df[column_map["latitude"]], errors="coerce")
        out["longitude"] = pd.to_numeric(raw_df[column_map["longitude"]], errors="coerce")
    else:
        out["latitude"] = np.nan
        out["longitude"] = np.nan

    valid_counts = out["pH"].notna().astype(int)
    risk_score = ((out["pH"] < 6.5) | (out["pH"] > 8.5)).fillna(False).astype(int)
    out["Risk_Score"] = risk_score.astype(int)
    out["Tested_Parameters"] = valid_counts.astype(int)
    out["Missing_Parameters"] = (1 - valid_counts).astype(int)
    out["Violated_Parameters"] = np.where(out["Risk_Score"] > 0, "pH", "")
    out["Risk_Level"] = np.select(
        [valid_counts == 0, out["Risk_Score"] > 0],
        ["Unknown", "Unsafe"],
        default="Safe",
    )
    return out, column_map, []


def build_uploaded_district_risk_frame(risk_df):
    valid = risk_df.dropna(subset=["District"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=[
            "District", "pH", "Risk_Level", "Risk_Score", "Tested_Parameters",
            "Violated_Parameters", "latitude", "longitude", "Samples",
        ])
    grouped = valid.groupby("District", as_index=False).agg(
        pH=("pH", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        Samples=("Risk_Level", "size"),
        Tested_Parameters=("Tested_Parameters", "sum"),
    )
    grouped["Risk_Score"] = ((grouped["pH"] < 6.5) | (grouped["pH"] > 8.5)).fillna(False).astype(int)
    grouped["Violated_Parameters"] = np.where(grouped["Risk_Score"] > 0, "pH", "")
    grouped["Risk_Level"] = np.select(
        [grouped["pH"].isna(), grouped["Risk_Score"] > 0],
        ["Unknown", "Unsafe"],
        default="Safe",
    )
    needs_coords = grouped["latitude"].isna() | grouped["longitude"].isna()
    if needs_coords.any():
        mapped = grouped.loc[needs_coords, "District"].map(
            lambda name: DISTRICT_COORDINATES.get(normalize_geo_name(name))
        )
        grouped.loc[needs_coords, "latitude"] = mapped.map(
            lambda coords: coords[0] if isinstance(coords, tuple) else np.nan
        )
        grouped.loc[needs_coords, "longitude"] = mapped.map(
            lambda coords: coords[1] if isinstance(coords, tuple) else np.nan
        )
    return grouped


# ─── SECTION: Data & Model Loading ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Loading CWC Water Quality Data...")
def load_and_preprocess_data():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    try:
        raw = load_all_csvs()
        source = "real"
    except FileNotFoundError:
        st.warning("Real data not found. Using synthetic data.")
        raw = generate_synthetic_cwc(n=15000)
        source = "synthetic"
    except Exception as e:
        st.error(f"Error loading data: {e}. Using synthetic.")
        raw = generate_synthetic_cwc(n=15000)
        source = "synthetic"
    
    return preprocess(raw), source


@st.cache_resource
def load_model():
    from huggingface_hub import hf_hub_download
    import joblib

    path = hf_hub_download(
        repo_id="crushh3/aquaintel-data",
        filename="models/rf_full.pkl",
        repo_type="dataset"
    )

    loaded = joblib.load(path)

    # 🔥 Handle dict structure
    if isinstance(loaded, dict):
        model = loaded["model"]
        features = loaded["features"]
    else:
        model = loaded

        if hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        elif hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "feature_names_in_"):
                    features = list(step.feature_names_in_)
                    break
        else:
            raise ValueError("Could not extract feature names from model")

    return model, features

if "df_base" not in st.session_state:
    st.session_state.df_base, st.session_state.data_source = load_and_preprocess_data()

df = st.session_state.df_base.copy()
source = st.session_state.data_source
model, features = load_model()

# ─── SECTION: Sidebar ───────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:1.2rem 0 0.8rem 0;border-bottom:1px solid rgba(255,255,255,0.1);margin-bottom:1rem;">
  <div style="font-size:1.4rem;font-weight:800;color:#ffffff;letter-spacing:-0.5px;">AquaIntel</div>
  <div style="font-size:0.72rem;color:#CAF0F8;margin-top:2px;font-weight:700;letter-spacing:0.05em;">
    WATER QUALITY PORTAL
  </div>
</div>
""", unsafe_allow_html=True)

# ── Dark / Light mode toggle ────────────────────────────────────────────────────
toggle_label = "Switch to Light Mode" if st.session_state.dark_mode else "Switch to Dark Mode"
if st.sidebar.button(toggle_label, use_container_width=True, key="theme_toggle"):
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()

st.sidebar.info("Filters apply to Dashboard, Historical Trends, and Parameter Analysis tabs only.")

with st.sidebar.expander("What is WQI?"):
    st.markdown("""
**Water Quality Index (WQI)** is a score from **0 to 100**.

**Lower score = Cleaner water.**

| Score   | Category      |
|---------|---------------|
| 0–25    | Excellent  |
| 25–50   | Good       |
| 50–75   | Poor       |
| 75–100  | Very Poor  |

*Standard: BIS 10500:2012 (Indian Drinking Water Standard)*
""")

st.sidebar.markdown("---")

# ── SEARCH SECTION ──────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<p style="color:#CAF0F8;font-size:0.72rem;letter-spacing:0.08em;'
    'font-weight:600;text-transform:uppercase;margin-bottom:4px;">LOCATION SEARCH</p>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Search by District, Tehsil, Block, Village, or River to get water quality info.")

search_query = st.sidebar.text_input(
    "Search Location", placeholder="Enter District, River, Village...", label_visibility="collapsed",
    key="location_search_input")

if search_query and search_query.strip():
    q = search_query.strip().lower()
    # Identify which columns to search across
    search_cols = []
    for c in ["district", "District", "tehsil", "Tehsil", "block", "Block",
               "village", "Village", "River", "river_name", "station_name", "station", "state"]:
        if c in df.columns:
            search_cols.append(c)

    if search_cols:
        mask = pd.Series([False] * len(df), index=df.index)
        for col in search_cols:
            mask = mask | df[col].astype(str).str.lower().str.contains(q, na=False)
        search_results = df[mask]
    else:
        search_results = pd.DataFrame()

    if not search_results.empty:
        avg_ph  = search_results["pH"].mean() if "pH" in search_results.columns else None
        avg_wqi = search_results["WQI"].mean() if "WQI" in search_results.columns else None
        safe_pct_s = (search_results["is_safe"].mean() * 100) if "is_safe" in search_results.columns else None
        n_records = len(search_results)

        # Verdict
        if avg_wqi is not None:
            if avg_wqi < 50:
                verdict_s, v_color = "SAFE", "#ffffff"
                v_bg = "#000000"
            else:
                verdict_s, v_color = "NOT SAFE", "#dc2626"
                v_bg = "#fee2e2"
        else:
            verdict_s, v_color, v_bg = "UNKNOWN", "#94a3b8", "#f1f5f9"

        ph_str  = f"{avg_ph:.2f}" if avg_ph is not None else "N/A"
        wqi_str = f"{avg_wqi:.1f}" if avg_wqi is not None else "N/A"
        sp_str  = f"{safe_pct_s:.1f}%" if safe_pct_s is not None else "N/A"

        st.sidebar.markdown(f"""
<div style="background:#1e3a5f;border:1px solid rgba(255,255,255,0.15);border-radius:10px;
            padding:0.9rem 1rem;margin-top:0.5rem;">
  <div style="font-size:0.7rem;color:#CAF0F8;font-weight:700;text-transform:uppercase;
              letter-spacing:0.05em;margin-bottom:6px;">Search Result — {n_records} records</div>
  <div style="background:{v_bg};border-radius:6px;padding:6px 10px;margin-bottom:8px;
              text-align:center;">
    <span style="font-size:1rem;font-weight:800;color:{v_color};">{verdict_s}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
    <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:6px 8px;">
      <div style="font-size:0.65rem;color:#CAF0F8;font-weight:700;">pH</div>
      <div style="font-size:1rem;font-weight:700;color:#f1f5f9;">{ph_str}</div>
    </div>
    <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:6px 8px;">
      <div style="font-size:0.65rem;color:#CAF0F8;font-weight:700;">Avg WQI</div>
      <div style="font-size:1rem;font-weight:700;color:#f1f5f9;">{wqi_str}</div>
    </div>
    <div style="background:rgba(255,255,255,0.05);border-radius:6px;padding:6px 8px;
                grid-column:1/-1;">
      <div style="font-size:0.65rem;color:#CAF0F8;font-weight:700;">Safe Samples</div>
      <div style="font-size:1rem;font-weight:700;color:#f1f5f9;">{sp_str}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.sidebar.warning(f"No records found for '{search_query}'. Try a different location name.")

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
    '<p style="color:#CAF0F8;font-size:0.72rem;letter-spacing:0.08em;'
    'font-weight:600;text-transform:uppercase;margin:0.8rem 0 4px 0;">GEOGRAPHICAL FILTER</p>',
    unsafe_allow_html=True,
)
sel_states = st.sidebar.multiselect(
    "States", states, key="state_filter", label_visibility="collapsed")

# Year range
st.sidebar.markdown(
    '<p style="color:#CAF0F8;font-size:0.72rem;letter-spacing:0.08em;'
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
    '<p style="color:#CAF0F8;font-size:0.72rem;letter-spacing:0.08em;'
    'font-weight:600;text-transform:uppercase;margin:0.8rem 0 4px 0;">WQI SCORE RANGE</p>',
    unsafe_allow_html=True,
)
wqi_range = st.sidebar.slider(
    "WQI Range", 0.0, 100.0, (0.0, 100.0),
    key="wqi_filter", label_visibility="collapsed")

st.sidebar.markdown("---")

st.sidebar.markdown('<div class="reset-btn-wrap">', unsafe_allow_html=True)
if st.sidebar.button("Reset Dashboard", use_container_width=True, key="reset_btn"):
    for k in ["state_filter", "year_filter", "wqi_filter"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Override reset button color specifically
st.sidebar.markdown("""
<style>
[data-testid="stSidebar"] div.reset-btn-wrap button {
    background: #ffffff !important;
    border: 1px solid #fecaca !important;
    color: #061A2E !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] div.reset-btn-wrap button *,
[data-testid="stSidebar"] div.reset-btn-wrap button p,
[data-testid="stSidebar"] div.reset-btn-wrap button span,
[data-testid="stSidebar"] div.reset-btn-wrap button div {
    color: #061A2E !important;
    -webkit-text-fill-color: #061A2E !important;
}
[data-testid="stSidebar"] div.reset-btn-wrap button:hover {
    background-color: #F1FAFF !important;
    color: #061A2E !important;
}
</style>
""", unsafe_allow_html=True)

# ─── SECTION: Data Filtering ────────────────────────────────────────────────────
def apply_filters(base_df, sel_states, sel_years, wqi_range):

    filt = base_df.copy()
    
    if sel_states:
        filt = filt[filt["state"].isin(sel_states)]
    
    if sel_years and "year" in filt.columns:
        filt = filt[(filt["year"] >= sel_years[0]) & (filt["year"] <= sel_years[1])]
    
    # WQI filter using query (faster for large data)
    if wqi_range and not filt.empty:
        filt = filt.query(f"WQI >= {wqi_range[0]} and WQI <= {wqi_range[1]}")
    

    if not filt.empty:
        filt["water_quality"] = filt["water_quality"].astype(str).str.strip()
    
    return filt

filt = apply_filters(df, sel_states, sel_years, wqi_range)

# ─── SECTION: App Header ────────────────────────────────────────────────────────
_dark = st.session_state.get("dark_mode", False)
_head_txt = "#f1f5f9" if _dark else "#0f172a"
st.markdown(
    f'<h1 style="color:{_head_txt};font-weight:800;font-size:1.75rem;margin:0 0 0.45rem 0;line-height:1.15;">'
    '💧AquaIntel Analytics</h1>',
    unsafe_allow_html=True,
)
st.markdown(f"""
<div style="background:#1e40af;color:#ffffff;padding:0.65rem 1.2rem;border-radius:8px;
            font-size:0.82rem;font-weight:500;margin-bottom:0.9rem;letter-spacing:0.01em;">
  AquaIntel Analytics — Water Quality Monitoring System &nbsp;|&nbsp;
  CWC Data: 1961–2020 &nbsp;|&nbsp; Standard: BIS 10500:2012 &nbsp;|&nbsp;
  SDG Goal 6: Clean Water &amp; Sanitation &nbsp;|&nbsp;
  <strong>{len(filt):,}</strong> records loaded &nbsp;·&nbsp; Source: {source}
</div>
""", unsafe_allow_html=True)

# ─── SECTION: Tab Layout ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard",
    " Risk Heatmap",
    " Historical Trends",
    " Parameter Analysis",
    " Predict",
    " Upload Data",
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

            with st.expander(" View Data Table"):
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
        with st.expander(" River Data Table"):
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
        with st.expander(" Monthly Data Table"):
            st.dataframe(monthly[["Month","WQI"]].rename(columns={"WQI":"Avg WQI"}).round(2),
                         use_container_width=True)



    # ── 3I: Help & Glossary ────────────────────────────────────────────────────────
    with st.expander("Help & Glossary of Terms"):
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
        with st.expander(" View Data Table"):
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
     Key Insights
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
            with st.expander(" View Data Table"):
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
            _dark4 = st.session_state.get("dark_mode", False)
            _sc_bg  = "#1e293b" if _dark4 else "#ffffff"
            _sc_brd = "#334155" if _dark4 else "#e2e8f0"
            _sc_lbl = "#f1f5f9" if _dark4 else "#0f172a"
            _sc_sub = "#94a3b8"
            for i, param in enumerate(avail_bis2):
                pct_exceed  = pct_exceeds(filt[param], BIS_STANDARDS[param])
                compliance  = 100 - pct_exceed if pd.notna(pct_exceed) else 0.0
                badge, bcolor, bbg = compliance_badge(compliance)
                bis_val = BIS_STANDARDS[param]
                bis_str = f"Max: {bis_val.get('max', bis_val.get('min','?'))}"
                label   = PARAM_LABELS.get(param, param)
                with cols_sc[i % 3]:
                    st.markdown(f"""
<div style="background:{_sc_bg};border:1px solid {_sc_brd};border-radius:10px;
            padding:1rem;margin-bottom:0.8rem;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
  <div style="font-size:0.78rem;font-weight:600;color:{_sc_lbl};">{label}</div>
  <div style="font-size:0.7rem;color:{_sc_sub};margin:2px 0;">BIS Limit — {bis_str}</div>
  <div style="display:flex;justify-content:space-between;align-items:center;margin-top:6px;">
    <div style="font-size:1.4rem;font-weight:800;color:{bcolor};">{compliance:.0f}%</div>
    <span style="background:{bbg};color:{bcolor};font-size:0.65rem;font-weight:700;
                 padding:2px 8px;border-radius:20px;">{badge}</span>
  </div>
  <div style="background:#334155;border-radius:4px;height:6px;margin-top:8px;">
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
                    f" **Statistical Relationship:** {sel_a_label} and {sel_b_label} show "
                    f"**{strength} {direction} correlation** (r = {r:.2f}). "
                    f"{'Higher values of both parameters tend to co-occur.' if r > 0.4 else ('Higher X values associate with lower Y values.' if r < -0.4 else 'No strong linear relationship detected.')}"
                )
            except Exception:
                pass

            with st.expander(" View Data Table"):
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

                with st.expander(" View Data Table"):
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
                    with st.expander("View Data Table"):
                        st.dataframe(exc_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Predict Tab (AI Water Quality Predictor + Location Safety Analysis)
# ═══════════════════════════════════════════════════════════════════════════════
# Default realistic water quality values (typical for a moderately polluted source)
PREDICT_DEFAULTS = {
    "dissolved_oxygen":  6.8,
    "total_hardness":  180.0,
    "BOD":               2.1,
    "COD":              12.0,
    "TDS":             320.0,
    "TSS":              18.0,
    "nitrates":          8.5,
    "ammonia":           0.3,
    "phosphate":         0.4,
    "chloride":         95.0,
    "fluoride":          0.7,
    "sulphate":         85.0,
    "conductivity":    480.0,
    "turbidity":         3.2,
    "total_coliform":   12.0,
    "fecal_coliform":    1.0,
    "arsenic":           0.005,
    "lead":              0.006,
    "iron":              0.2,
    "manganese":         0.05,
    "pH":                7.2,
    "temperature":      27.0,
    "nitrites":          0.05,
    "calcium":          52.0,
    "magnesium":        18.0,
}

with tab5:
    _dark5 = st.session_state.get("dark_mode", False)
    _t5col = "#f1f5f9" if _dark5 else "#0f172a"

    # ── AI Predictor ─────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:1.5rem;font-weight:800;color:{_t5col};margin-bottom:0.5rem;">'
        'AI Water Quality Predictor</div>',
        unsafe_allow_html=True,
    )
    st.caption("Enter parameter values below to predict water safety using trained ML models. "
               "Default values represent a realistic moderate-quality water sample.")

    if model:
        model_rf = model
        inputs = {}
        exclude_ui = [
               "is_safe",
                "water_quality",
                "wqi",

                # metadata / useless
                "slno",
                "state lgd code",
                "district lgd code",
                "latitude",
                "longitude",
                "WQI",

                # duplicate / noisy
                "nitrate n (mgn/l)"
            ]

        features = [f for f in features if f not in exclude_ui]

        with st.container(border=True):
            cols_p = st.columns(3)

            for i, feat in enumerate(features):
                default_val = float(PREDICT_DEFAULTS.get(feat, 0.0))

                inputs[feat] = cols_p[i % 3].number_input(
                    PARAM_LABELS.get(feat, feat),
                    value=default_val,
                    key=f"pred_{feat}",
                )

            predict_clicked = st.button(
                "Generate Prediction",
                use_container_width=True,
                key="predict_main_btn"
            )

        if predict_clicked:
            input_df = pd.DataFrame([inputs])[features]

            rf_pred = model_rf.predict(input_df)[0]

            if rf_pred == 1:
                st.success("Your Water is Safe For Drinking")
            else:
                st.error("Your Water is NOT Safe For Drinking")

            st.markdown("---")

            pred_summary = {
                "Random Forest (Main Model)": "Safe" if rf_pred == 1 else "Unsafe"
            }

            st.dataframe(pd.DataFrame(list(pred_summary.items()), columns=["Model", "Prediction"]))

    else:
        st.warning("Model not loaded. Please check Hugging Face model path.")

    # ── Lite Prediction ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div style="font-size:1.5rem;font-weight:800;color:{_t5col};margin-bottom:0.5rem;">'
        'Lite Water Quality Prediction</div>',
        unsafe_allow_html=True,
    )
    st.caption("Quick estimation using pH, Conductivity, and Nitrates (for demo/educational use)")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        ph   = col1.number_input("pH", 0.0, 14.0, 7.2, key="lite_ph")
        cond = col2.number_input("Conductivity (µS/cm)", 0.0, 2000.0, 480.0, key="lite_cond")
        nit  = col3.number_input("Nitrates (mg/L)", 0.0, 100.0, 8.5, key="lite_nit")

        lite_clicked = st.button("Run Lite Prediction", key="lite_btn", use_container_width=True)

    if lite_clicked:
        score = 0
        if ph < 6.5 or ph > 8.5: score += 1
        if cond > 750:            score += 1
        if nit > 45:              score += 1

        if score == 0:
            st.success("Your Water is Safe For Drinking")
        elif score == 1:
            st.warning("Your Water is Moderately Safe For Drinking")
        else:
            st.error("Your Water is NOT Safe For Drinking and Treatment required")

    # ── Location Safety Analysis ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f'<div style="font-size:1.5rem;font-weight:800;color:{_t5col};margin-bottom:1rem;">'
        'Location-based Safety Analysis</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        col1l, col2l = st.columns(2)
        user_lat = col1l.number_input("Enter Latitude",  value=12.97, key="loc_lat")
        user_lon = col2l.number_input("Enter Longitude", value=77.59, key="loc_lon")
        analyze_clicked = st.button("Analyze Current Location", use_container_width=True, key="loc_analyze_btn")

    if analyze_clicked:
        if model is None or features is None:
            st.error("Model not loaded.")
        else:
            st.markdown("**Nearest Monitoring Stations**")

            weighted_sum = 0
            total_weight = 0
            results = []
            filt_unique = filt.drop_duplicates(subset=["latitude", "longitude"])
            nearest = get_nearest_stations(filt_unique, user_lat, user_lon, n=3)

            if nearest.empty:
                st.error("No station data available.")
            else:
                for _, row in nearest.iterrows():

                    # build input using model features
                    input_row = {f: row.get(f, 0) for f in features}

                    input_df = pd.DataFrame([input_row])

                    # ensure all expected columns exist
                    for f in model.feature_names_in_:
                        if f not in input_df.columns:
                            input_df[f] = 0

                    input_df = input_df[model.feature_names_in_]

                    try:
                        pred = model.predict(input_df)[0]
                        proba = model.predict_proba(input_df)[0][1]
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        continue

                    distance = row["distance_km"]

                    weight = np.exp(-distance / 50)
                    weighted_sum += proba * weight
                    total_weight += weight

                    results.append({
                        "Station": row.get("station", "Unknown"),
                        "Distance (km)": round(distance, 2),
                        "Prediction": "SAFE" if pred == 1 else "UNSAFE",
                        "Confidence": round(proba, 3),
                        "latitude": row["latitude"],
                        "longitude": row["longitude"],
                    })

                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    # Ensure numeric
                    results_df["latitude"] = pd.to_numeric(results_df["latitude"], errors="coerce")
                    results_df["longitude"] = pd.to_numeric(results_df["longitude"], errors="coerce")

                    results_df = results_df.dropna(subset=["latitude", "longitude"])

                    if not results_df.empty:

                        # 🔥 Force visible marker size + colors
                        fig_loc = px.scatter_mapbox(
                            results_df,
                            lat="latitude",
                            lon="longitude",
                            color="Prediction",
                            zoom=10,
                            center={"lat": user_lat, "lon": user_lon},
                            mapbox_style="carto-positron",
                            hover_name="Station",
                            hover_data={
                                "Distance (km)": True,
                                "Confidence": True,
                                "latitude": False,
                                "longitude": False
                            },
                            color_discrete_map={
                                "SAFE": "#00FF7F",   # bright green
                                "UNSAFE": "#FF4C4C"  # bright red
                            }
                        )

                        # 🔥 CRITICAL: force marker visibility
                        fig_loc.update_traces(
                            marker=dict(size=16, opacity=0.95),
                            selector=dict(mode="markers")
                        )

                        # 🔥 Add user location marker (always visible)
                        fig_loc.add_scattermapbox(
                            lat=[user_lat],
                            lon=[user_lon],
                            mode="markers",
                            marker=dict(size=18, color="black", symbol="star"),
                            name="You"
                        )

                        # 🔥 Layout fix (prevents clipping)
                        fig_loc.update_layout(
                            height=500,
                            margin=dict(l=0, r=0, t=20, b=0)
                        )

                        st.plotly_chart(fig_loc, use_container_width=True)

                    else:
                        st.warning("No valid coordinates to display on map.")

                    final_score = weighted_sum / total_weight if total_weight > 0 else 0

                    st.markdown("---")

                    if final_score > 0.5:
                        st.success(f"✅ Likely SAFE ({final_score:.1%})")
                    else:
                        st.error(f"🚨 Likely UNSAFE ({final_score:.1%})")
                else:
                    st.error("Model/Features not loaded correctly. Check your HF Hub connection.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION: Upload Data Tab
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("""
<div class="risk-hero">
  <h3>Upload Water Risk Dataset</h3>
  <p>Analyze uploaded CSV or Excel files using pH safety limits and district coordinates.</p>
</div>
""", unsafe_allow_html=True)
    st.info(
        "Upload a CSV or Excel file. Required fields are District/Location and pH. "
        "Latitude/longitude are used to map districts when available."
    )
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded:
        with st.spinner("Processing uploaded file..."):
            new_df = read_uploaded_water_file(uploaded)
            st.success(f"Loaded {len(new_df):,} rows x {len(new_df.columns)} columns")
            risk_df, upload_column_map, validation_errors = prepare_uploaded_risk_data(new_df)

        if validation_errors:
            if "Uploaded file has no rows." in validation_errors:
                st.error("Uploaded file has no rows.")
            else:
                st.error(
                    "Required columns missing: "
                    + ", ".join(label.title() for label in validation_errors)
                    + ". Please include District/Location and pH."
                )
                with st.expander("Detected upload columns"):
                    detected = {
                        key: (value if value else "Not found")
                        for key, value in upload_column_map.items()
                    }
                    st.dataframe(pd.DataFrame([detected]), use_container_width=True)
        else:
            new_df["Risk_Level"] = risk_df["Risk_Level"].values
            detected_cols = {
                "District / Location": upload_column_map.get("district"),
                "pH": upload_column_map.get("pH"),
                "Latitude": upload_column_map.get("latitude") or "Fallback mapping",
                "Longitude": upload_column_map.get("longitude") or "Fallback mapping",
            }
            with st.expander("Column detection details"):
                st.dataframe(pd.DataFrame([detected_cols]), use_container_width=True)

            district_risk_df = build_uploaded_district_risk_frame(risk_df)
            unknown_rows = int((risk_df["Tested_Parameters"] == 0).sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg pH", "N/A" if pd.isna(risk_df["pH"].mean()) else f"{risk_df['pH'].mean():.2f}")
            m2.metric("Safe Districts", int((district_risk_df["Risk_Level"] == "Safe").sum()) if not district_risk_df.empty else 0)
            m3.metric("Unsafe Districts", int((district_risk_df["Risk_Level"] == "Unsafe").sum()) if not district_risk_df.empty else 0)
            m4.metric("Total Districts", int(district_risk_df["District"].nunique()) if not district_risk_df.empty else 0)

            if unknown_rows:
                st.warning(
                    f"{unknown_rows:,} rows have no usable recognized parameter values and are marked as Unknown."
                )

            filter_col1, filter_col2 = st.columns([1, 2])
            with filter_col1:
                risk_filter = st.multiselect(
                    "Risk Level",
                    ["Safe", "Unsafe", "Unknown"],
                    default=["Safe", "Unsafe", "Unknown"],
                    key="uploaded_risk_filter",
                )
            with filter_col2:
                district_search = st.text_input(
                    "District Search",
                    placeholder="Type a district or location name...",
                    key="uploaded_district_search",
                )

            filtered_districts = district_risk_df.copy()
            if risk_filter:
                filtered_districts = filtered_districts[filtered_districts["Risk_Level"].isin(risk_filter)]
            if district_search.strip():
                q = district_search.strip().lower()
                filtered_districts = filtered_districts[
                    filtered_districts["District"].astype(str).str.lower().str.contains(q, na=False)
                ]

            st.markdown("""
<div class="risk-legend">
  <span><i class="risk-dot" style="background:#16a34a;"></i>Safe</span>
  <span><i class="risk-dot" style="background:#dc2626;"></i>Unsafe</span>
  <span><i class="risk-dot" style="background:#94a3b8;"></i>Unknown</span>
</div>
""", unsafe_allow_html=True)

            map_df = filtered_districts.dropna(subset=["latitude", "longitude"]).copy()
            no_coord_count = len(filtered_districts) - len(map_df)
            if no_coord_count > 0:
                st.warning(
                    f"Coordinates unavailable for {no_coord_count:,} district(s). "
                    "Add latitude/longitude columns or extend the predefined district coordinate mapping."
                )

            if map_df.empty:
                st.warning("No mappable district records after filters. Check coordinates or filter selection.")
            else:
                risk_colors = {"Safe": "#16a34a", "Unsafe": "#dc2626", "Unknown": "#94a3b8"}
                fig_risk_map = px.scatter_mapbox(
                    map_df,
                    lat="latitude",
                    lon="longitude",
                    color="Risk_Level",
                    color_discrete_map=risk_colors,
                    size="Samples",
                    size_max=24,
                    center={"lat": float(map_df["latitude"].mean()), "lon": float(map_df["longitude"].mean())},
                    zoom=5 if len(map_df) <= 40 else 4,
                    mapbox_style="carto-positron",
                    hover_name="District",
                    hover_data={
                        "Risk_Level": True,
                        "pH": ":.2f",
                        "Risk_Score": True,
                        "Tested_Parameters": True,
                        "Violated_Parameters": True,
                        "Samples": True,
                        "latitude": False,
                        "longitude": False,
                    },
                    category_orders={"Risk_Level": ["Safe", "Unsafe", "Unknown"]},
                    title="District-Level pH Safety Risk Map",
                )
                fig_risk_map.update_traces(marker=dict(opacity=0.88))
                fig_risk_map.update_layout(
                    height=620,
                    margin=dict(l=0, r=0, t=42, b=0),
                    paper_bgcolor="#F1FAFF",
                    legend=dict(
                        title="Risk Level",
                        orientation="h",
                        yanchor="bottom",
                        y=1.01,
                        xanchor="right",
                        x=1,
                    ),
                )
                st.plotly_chart(
                    fig_risk_map,
                    use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False},
                )

            c1, c2 = st.columns([1.25, 1])
            with c1:
                st.markdown("#### District Risk Summary")
                summary_cols = [
                    "District", "Risk_Level", "pH", "Risk_Score", "Tested_Parameters",
                    "Violated_Parameters", "Samples", "latitude", "longitude",
                ]
                st.dataframe(
                    filtered_districts[summary_cols].sort_values(["Risk_Level", "District"]).round(3),
                    use_container_width=True,
                    height=360,
                )
            with c2:
                risk_counts_df = (
                    risk_df["Risk_Level"]
                    .value_counts()
                    .reindex(["Safe", "Unsafe", "Unknown"])
                    .fillna(0)
                    .astype(int)
                    .rename_axis("Risk_Level")
                    .reset_index(name="Records")
                )
                fig_counts = px.bar(
                    risk_counts_df,
                    x="Risk_Level",
                    y="Records",
                    color="Risk_Level",
                    color_discrete_map={"Safe": "#16a34a", "Unsafe": "#dc2626", "Unknown": "#94a3b8"},
                    labels={"Risk_Level": "Risk Level", "Records": "Records"},
                    title="Uploaded Record Risk Distribution",
                )
                fig_counts = update_chart_layout(fig_counts, height=360)
                fig_counts.update_layout(showlegend=False)
                st.plotly_chart(fig_counts, use_container_width=True, config={"displayModeBar": False})

            with st.expander("Preview processed upload with Risk_Level"):
                st.dataframe(risk_df.head(1000), use_container_width=True, height=360)

            st.download_button(
                "Download Processed Risk Dataset",
                data=risk_df.to_csv(index=False),
                file_name=f"water_risk_upload_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_uploaded_risk_csv",
            )

        if "WQI" in new_df.columns:
            fig_up = px.histogram(
                new_df, x="WQI", nbins=40,
                title="WQI Distribution in Uploaded Data",
                labels={"WQI": "Water Quality Index (WQI)"},
                color_discrete_sequence=["#1e40af"],
            )
            fig_up.add_vline(x=50, line_dash="dash", line_color="#dc2626",
                annotation_text="BIS Safety Threshold (WQI = 50)")
            fig_up = update_chart_layout(fig_up, height=420)
            fig_up.update_layout(xaxis_title="Water Quality Index (WQI) — Lower is Better",
                                 yaxis_title="Number of Records")
            st.plotly_chart(fig_up, use_container_width=True)

            with st.expander("Summary Statistics"):
                st.dataframe(new_df.describe().round(2), use_container_width=True)
        else:
            st.caption("No WQI column found; upload risk analysis above uses pH only.")
