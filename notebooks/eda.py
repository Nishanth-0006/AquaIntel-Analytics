"""
AquaIntel Analytics — Exploratory Data Analysis
Run: python eda.py
Outputs all figures to /figures/
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from utils.data_loader import (
    load_all_csvs, generate_synthetic_cwc, preprocess,
    CORE_FEATURES, BIS_STANDARDS, STATE_CODES
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── colour palette ──────────────────────────────────────────────────────────
PALETTE     = ["#1f4e79", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d", "#3b1f2b"]
QUAL_COLORS = {"Excellent": "#2ecc71", "Good": "#f1c40f",
               "Poor": "#e67e22", "Very Poor": "#e74c3c"}

sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)

# ─── load data ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
try:
    raw = load_all_csvs(DATA_DIR)
    print(f"✅  Loaded real data: {raw.shape}")
except FileNotFoundError:
    print("⚠️  No CSVs found in /data — using synthetic demo data (5 000 rows).")
    raw = generate_synthetic_cwc(n=5000)

df = preprocess(raw)
print(f"    After preprocessing: {df.shape}")
print(f"    States: {df['state'].unique().tolist()}")
print(f"    Years : {int(df['year'].min()) if 'year' in df.columns else 'N/A'} – "
      f"{int(df['year'].max()) if 'year' in df.columns else 'N/A'}")
print(f"    WQI   : {df['WQI'].describe().to_dict()}\n")

avail_features = [f for f in CORE_FEATURES if f in df.columns]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dataset Overview
# ═══════════════════════════════════════════════════════════════════════════
def fig1_overview():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("EDA – Fig 1: Dataset Overview", fontsize=14, fontweight="bold")

    # (a) Records per state
    state_counts = df["state"].value_counts()
    axes[0].barh(state_counts.index, state_counts.values, color=PALETTE[1])
    axes[0].set_title("Records per State")
    axes[0].set_xlabel("Count")
    for i, v in enumerate(state_counts.values):
        axes[0].text(v + 5, i, str(v), va="center", fontsize=9)

    # (b) Year distribution
    if "year" in df.columns:
        df["year"].dropna().astype(int).value_counts().sort_index().plot(
            kind="bar", ax=axes[1], color=PALETTE[0], edgecolor="white", width=0.8)
        axes[1].set_title("Observations by Year")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=45, labelsize=7)
    else:
        axes[1].text(0.5, 0.5, "No year column", ha="center", va="center")

    # (c) Water quality distribution
    wq_counts = df["water_quality"].value_counts()
    colors = [QUAL_COLORS.get(str(c), "#999") for c in wq_counts.index]
    wedges, texts, autotexts = axes[2].pie(
        wq_counts.values, labels=wq_counts.index,
        autopct="%1.1f%%", colors=colors, startangle=140)
    axes[2].set_title("Water Quality Distribution (WQI)")

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig1_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  fig1_overview.png")

fig1_overview()


# ═══════════════════════════════════════════════════════════════════════════
# 2. WQI Distribution & Trends
# ═══════════════════════════════════════════════════════════════════════════
def fig2_wqi():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("EDA – Fig 2: WQI Analysis", fontsize=14, fontweight="bold")

    # (a) WQI histogram
    axes[0].hist(df["WQI"].dropna(), bins=40, color=PALETTE[1],
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(50, color="red", ls="--", lw=1.5, label="Safe threshold (50)")
    axes[0].set_title("WQI Distribution")
    axes[0].set_xlabel("WQI Score")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # (b) WQI by state
    state_wqi = df.groupby("state")["WQI"].median().sort_values()
    colors = [QUAL_COLORS["Good"] if v <= 50 else QUAL_COLORS["Poor"]
              for v in state_wqi.values]
    axes[1].barh(state_wqi.index, state_wqi.values, color=colors)
    axes[1].axvline(50, color="red", ls="--", lw=1.5)
    axes[1].set_title("Median WQI by State")
    axes[1].set_xlabel("Median WQI")

    # (c) WQI trend over years
    if "year" in df.columns:
        trend = df.groupby("year")["WQI"].median().reset_index()
        axes[2].plot(trend["year"], trend["WQI"],
                     marker="o", ms=4, color=PALETTE[0], lw=2)
        axes[2].axhline(50, color="red", ls="--", lw=1.5, label="Safe threshold")
        axes[2].fill_between(trend["year"], trend["WQI"], 50,
                             where=(trend["WQI"] > 50), alpha=0.2, color="red")
        axes[2].fill_between(trend["year"], trend["WQI"], 50,
                             where=(trend["WQI"] <= 50), alpha=0.2, color="green")
        axes[2].set_title("Median WQI Trend Over Years")
        axes[2].set_xlabel("Year")
        axes[2].set_ylabel("Median WQI")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No year column", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig2_wqi.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  fig2_wqi.png")

fig2_wqi()


# ═══════════════════════════════════════════════════════════════════════════
# 3. BIS Exceedance Analysis
# ═══════════════════════════════════════════════════════════════════════════
def fig3_bis_exceedance():
    exceedance = {}
    for param, std in BIS_STANDARDS.items():
        if param not in df.columns:
            continue
        col = df[param].dropna()
        if "max" in std:
            pct = (col > std["max"]).mean() * 100
        elif "min" in std:
            pct = (col < std["min"]).mean() * 100
        else:
            lo, hi = std["min"], std["max"]
            pct = ((col < lo) | (col > hi)).mean() * 100
        exceedance[param] = round(pct, 1)

    exc_s = pd.Series(exceedance).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("EDA – Fig 3: BIS Standard Exceedance", fontsize=14, fontweight="bold")

    # (a) Bar chart
    colors = ["#e74c3c" if v > 30 else "#e67e22" if v > 10 else "#2ecc71"
              for v in exc_s.values]
    axes[0].barh(exc_s.index, exc_s.values, color=colors)
    axes[0].axvline(20, color="gray", ls="--", lw=1, label="20% threshold")
    axes[0].set_title("% Samples Exceeding BIS Limits")
    axes[0].set_xlabel("Exceedance %")
    for i, v in enumerate(exc_s.values):
        axes[0].text(v + 0.3, i, f"{v}%", va="center", fontsize=8)

    # (b) Heatmap: exceedance by state
    state_exc = {}
    for param in exc_s.index:
        if param not in df.columns:
            continue
        std = BIS_STANDARDS[param]
        for state, grp in df.groupby("state"):
            col = grp[param].dropna()
            if len(col) == 0:
                continue
            if "max" in std:
                pct = (col > std["max"]).mean() * 100
            elif "min" in std:
                pct = (col < std["min"]).mean() * 100
            else:
                lo, hi = std["min"], std["max"]
                pct = ((col < lo) | (col > hi)).mean() * 100
            state_exc.setdefault(state, {})[param] = round(pct, 1)

    hm_df = pd.DataFrame(state_exc).T.fillna(0)
    hm_df = hm_df[[c for c in exc_s.index if c in hm_df.columns]]
    sns.heatmap(hm_df, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, ax=axes[1], cbar_kws={"label": "Exceedance %"})
    axes[1].set_title("Exceedance % by State × Parameter")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig3_bis_exceedance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  fig3_bis_exceedance.png")

fig3_bis_exceedance()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Parameter Distributions (violin + box)
# ═══════════════════════════════════════════════════════════════════════════
def fig4_distributions():
    show_params = [f for f in ["pH", "dissolved_oxygen", "BOD", "TDS",
                               "nitrates", "conductivity", "turbidity", "ammonia"]
                   if f in df.columns]
    if not show_params:
        print("⚠️  No params for fig4")
        return

    ncols = 4
    nrows = (len(show_params) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle("EDA – Fig 4: Parameter Distributions by State", fontsize=14, fontweight="bold")

    for ax, param in zip(axes, show_params):
        data = df[[param, "state"]].dropna()
        if data.empty:
            ax.set_visible(False)
            continue
        states = data["state"].unique()
        positions = range(len(states))
        parts = ax.violinplot([data[data["state"] == s][param].values for s in states],
                               positions=positions, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE[1])
            pc.set_alpha(0.6)
        ax.set_xticks(list(positions))
        ax.set_xticklabels([s[:6] for s in states], rotation=45, fontsize=7)
        ax.set_title(param)
        # BIS limit line
        std = BIS_STANDARDS.get(param, {})
        if "max" in std:
            ax.axhline(std["max"], color="red", ls="--", lw=1, label=f"BIS max={std['max']}")
            ax.legend(fontsize=7)
        elif "min" in std:
            ax.axhline(std["min"], color="green", ls="--", lw=1, label=f"BIS min={std['min']}")
            ax.legend(fontsize=7)

    for ax in axes[len(show_params):]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig4_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  fig4_distributions.png")

fig4_distributions()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig5_correlation():
    num_cols = [f for f in avail_features if f in df.columns] + ["WQI"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, linewidths=0.5,
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("EDA – Fig 5: Pearson Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig5_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  fig5_correlation.png")

fig5_correlation()


# ═══════════════════════════════════════════════════════════════════════════
# 6. Seasonal Analysis
# ═══════════════════════════════════════════════════════════════════════════
def fig6_seasonal():
    if "month" not in df.columns:
        print("⚠️  No month column — skipping fig6")
        return

    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    df["month_name"] = df["month"].map(month_names)

    show = [f for f in ["pH", "dissolved_oxygen", "BOD", "turbidity", "nitrates", "TDS"]
            if f in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    fig.suptitle("EDA – Fig 6: Seasonal Variation of Key Parameters", fontsize=14, fontweight="bold")

    months_order = [month_names[i] for i in range(1, 13)]

    for ax, param in zip(axes, show):
        monthly = (df.groupby("month")[param]
                     .agg(["median", lambda x: x.quantile(0.25),
                           lambda x: x.quantile(0.75)])
                     .reset_index())
        monthly.columns = ["month", "median", "q25", "q75"]
        monthly = monthly.sort_values("month")
        mnames = [month_names.get(int(m), m) for m in monthly["month"]]

        ax.plot(range(len(monthly)), monthly["median"], marker="o",
                ms=5, color=PALETTE[0], lw=2, label="Median")
        ax.fill_between(range(len(monthly)), monthly["q25"], monthly["q75"],
                        alpha=0.25, color=PALETTE[1], label="IQR")
        ax.set_xticks(range(len(monthly)))
        ax.set_xticklabels(mnames, rotation=45, fontsize=8)
        ax.set_title(param)
        ax.legend(fontsize=7)

        std = BIS_STANDARDS.get(param, {})
        if "max" in std:
            ax.axhline(std["max"], color="red", ls="--", lw=1)
        elif "min" in std:
            ax.axhline(std["min"], color="green", ls="--", lw=1)

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig6_seasonal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  fig6_seasonal.png")

fig6_seasonal()


# ═══════════════════════════════════════════════════════════════════════════
# 7. Missing Data Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig7_missing():
    check_cols = [f for f in avail_features if f in df.columns]
    missing_pct = (df[check_cols].isna().mean() * 100).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("EDA – Fig 7: Missing Data Analysis", fontsize=14, fontweight="bold")

    # Bar
    colors = ["#e74c3c" if v > 50 else "#e67e22" if v > 20 else "#2ecc71"
              for v in missing_pct.values]
    axes[0].barh(missing_pct.index, missing_pct.values, color=colors)
    axes[0].axvline(50, color="gray", ls="--", lw=1, label="50% threshold")
    axes[0].set_title("Missing Values per Parameter (%)")
    axes[0].set_xlabel("Missing %")
    for i, v in enumerate(missing_pct.values):
        axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)

    # Missing by state heatmap
    miss_state = df.groupby("state")[check_cols].apply(lambda x: x.isna().mean() * 100)
    sns.heatmap(miss_state, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.5, ax=axes[1], cbar_kws={"label": "Missing %"})
    axes[1].set_title("Missing % by State × Parameter")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig7_missing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  fig7_missing.png")

fig7_missing()


# ═══════════════════════════════════════════════════════════════════════════
# 8. Outlier Analysis (IQR method)
# ═══════════════════════════════════════════════════════════════════════════
def fig8_outliers():
    show = [f for f in ["pH", "conductivity", "TDS", "BOD", "nitrates",
                        "dissolved_oxygen", "turbidity", "ammonia"]
            if f in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    fig.suptitle("EDA – Fig 8: Outlier Detection (Box Plots by State)",
                 fontsize=14, fontweight="bold")

    for ax, param in zip(axes, show):
        data_groups = [df[df["state"] == s][param].dropna().values
                       for s in df["state"].unique()]
        labels = [s[:6] for s in df["state"].unique()]
        bp = ax.boxplot(data_groups, labels=labels, patch_artist=True,
                        medianprops={"color": "red"}, flierprops={"marker": ".", "ms": 3})
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE[1])
            patch.set_alpha(0.6)
        ax.set_title(param)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    for ax in axes[len(show):]:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig8_outliers.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  fig8_outliers.png")

fig8_outliers()


# ═══════════════════════════════════════════════════════════════════════════
# 9. Scatter Matrix (key params)
# ═══════════════════════════════════════════════════════════════════════════
def fig9_scatter():
    show = [f for f in ["pH", "dissolved_oxygen", "BOD", "TDS", "nitrates", "WQI"]
            if f in df.columns]
    if len(show) < 3:
        print("⚠️  Too few params for scatter matrix — skipping fig9")
        return

    sample = df[show + ["water_quality"]].dropna().sample(
        min(1000, len(df)), random_state=42)

    fig, axes = plt.subplots(len(show), len(show), figsize=(14, 12))
    fig.suptitle("EDA – Fig 9: Scatter Matrix (Key Parameters)",
                 fontsize=14, fontweight="bold")
    qual_list = ["Excellent", "Good", "Poor", "Very Poor"]

    for i, p1 in enumerate(show):
        for j, p2 in enumerate(show):
            ax = axes[i][j]
            if i == j:
                ax.hist(sample[p1].dropna(), bins=20,
                        color=PALETTE[1], edgecolor="white", alpha=0.8)
                ax.set_ylabel(p1, fontsize=8)
            else:
                for q in qual_list:
                    sub = sample[sample["water_quality"].astype(str) == q]
                    ax.scatter(sub[p2], sub[p1], s=5, alpha=0.4,
                               color=QUAL_COLORS.get(q, "#999"), label=q)
            ax.tick_params(labelsize=6)
            if i == len(show) - 1:
                ax.set_xlabel(p2, fontsize=8)

    handles = [plt.Line2D([0],[0], marker="o", color="w",
               markerfacecolor=QUAL_COLORS[q], ms=8, label=q) for q in qual_list]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig9_scatter_matrix.png", dpi=130, bbox_inches="tight")
    plt.close()
    print("✅  fig9_scatter_matrix.png")

fig9_scatter()


# ═══════════════════════════════════════════════════════════════════════════
# 10. Summary Stats Table
# ═══════════════════════════════════════════════════════════════════════════
def summary_stats():
    cols = [f for f in avail_features if f in df.columns] + ["WQI"]
    stats_df = df[cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T.round(3)
    stats_df["% missing"] = (df[cols].isna().mean() * 100).round(1)
    # BIS limit
    stats_df["BIS_limit"] = [
        BIS_STANDARDS.get(c, {}).get("max", BIS_STANDARDS.get(c, {}).get("min", "—"))
        for c in stats_df.index
    ]
    stats_df.to_csv(f"{FIGURES_DIR}/summary_stats.csv")
    print("✅  summary_stats.csv")
    print(stats_df.to_string())

summary_stats()

print("\n✅  EDA complete. All figures saved to", FIGURES_DIR)
