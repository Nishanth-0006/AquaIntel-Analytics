import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="AquaIntel Analytics Lite",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_PATH = "swq_manual_chemical_parameters_cwc_ap_1961_2020.csv"
PRIMARY = "#0b6e4f"
ACCENT = "#d62828"


@st.cache_data(show_spinner=False)
def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner=False)
def build_lite_dataset(path: str) -> pd.DataFrame:
    df = load_raw_data(path).copy()
    df["Nitrate (mg/L) Derived"] = df["Nitrate N (mgN/L)"] * 4.43
    df["Conductivity Proxy (uS/cm)"] = df["Total Dissolved Solids (mg/L)"] / 0.64

    feature_cols = [
        "Potential of Hydrogen (pH)",
        "Conductivity Proxy (uS/cm)",
        "Nitrate (mg/L) Derived",
    ]

    lite_df = df[
        [
            "State",
            "District",
            "Latitude",
            "Longitude",
            "Data Acquisition Time",
        ]
        + feature_cols
    ].dropna().copy()

    lite_df["timestamp"] = pd.to_datetime(
        lite_df["Data Acquisition Time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )
    lite_df["year"] = lite_df["timestamp"].dt.year
    lite_df["safe_flag"] = (
        lite_df["Potential of Hydrogen (pH)"].between(6.5, 8.5)
        & (lite_df["Conductivity Proxy (uS/cm)"] <= 1500)
        & (lite_df["Nitrate (mg/L) Derived"] <= 45)
    ).astype(int)
    lite_df["risk_label"] = np.where(lite_df["safe_flag"] == 1, "Safe", "At Risk")

    return lite_df


@st.cache_resource(show_spinner=False)
def train_baseline_model(path: str):
    lite_df = build_lite_dataset(path)
    feature_cols = [
        "Potential of Hydrogen (pH)",
        "Conductivity Proxy (uS/cm)",
        "Nitrate (mg/L) Derived",
    ]

    X = lite_df[feature_cols]
    y = lite_df["safe_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    metrics = {
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "holdout_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "holdout_auc": float(roc_auc_score(y_test, y_proba)),
        "cv_accuracy_mean": float(cv_accuracy.mean()),
        "cv_auc_mean": float(cv_auc.mean()),
    }

    return model, metrics


def score_sample(model: Pipeline, ph: float, conductivity: float, nitrates: float):
    sample = pd.DataFrame(
        [
            {
                "Potential of Hydrogen (pH)": ph,
                "Conductivity Proxy (uS/cm)": conductivity,
                "Nitrate (mg/L) Derived": nitrates,
            }
        ]
    )
    safe_probability = float(model.predict_proba(sample)[0, 1])
    ml_label = "Safe" if safe_probability >= 0.5 else "At Risk"
    rule_label = (
        "Safe"
        if (6.5 <= ph <= 8.5 and conductivity <= 1500 and nitrates <= 45)
        else "At Risk"
    )
    return safe_probability, ml_label, rule_label


raw_df = load_raw_data(DATA_PATH)
lite_df = build_lite_dataset(DATA_PATH)
model, model_metrics = train_baseline_model(DATA_PATH)

available_districts = sorted(lite_df["District"].dropna().unique().tolist())
min_year = int(lite_df["year"].dropna().min())
max_year = int(lite_df["year"].dropna().max())

st.title("AquaIntel Analytics", anchor=False)

with st.sidebar:
    st.header("Dashboard Controls", anchor=False)
    year_range = st.slider(
        "Year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )
    selected_districts = st.multiselect(
        "Districts",
        options=available_districts,
        default=available_districts,
    )
    selected_risk = st.multiselect(
        "Risk labels",
        options=["Safe", "At Risk"],
        default=["Safe", "At Risk"],
    )
    st.caption("Filters apply to the visual dashboard and data explorer below.")

filtered_df = lite_df[
    lite_df["District"].isin(selected_districts)
    & lite_df["risk_label"].isin(selected_risk)
    & lite_df["year"].between(year_range[0], year_range[1], inclusive="both")
].copy()

total_filtered = len(filtered_df)
safe_share = filtered_df["safe_flag"].mean() * 100 if total_filtered else 0.0
at_risk_share = 100 - safe_share if total_filtered else 0.0
district_count = filtered_df["District"].nunique() if total_filtered else 0
year_span_text = f"{year_range[0]}-{year_range[1]}"

metrics_row = st.columns(4)
with metrics_row[0]:
    st.metric("Filtered Samples", f"{total_filtered:,}")
with metrics_row[1]:
    st.metric("Safe Share", f"{safe_share:.1f}%")
with metrics_row[2]:
    st.metric("At-Risk Share", f"{at_risk_share:.1f}%")
with metrics_row[3]:
    st.metric("Coverage Window", year_span_text)

if filtered_df.empty:
    st.warning("No data matches the selected filters. Broaden the year range or district selection.")
    st.stop()

district_summary = (
    filtered_df.groupby("District", as_index=False)
    .agg(
        samples=("safe_flag", "size"),
        safe_share=("safe_flag", "mean"),
        median_ph=("Potential of Hydrogen (pH)", "median"),
        median_conductivity=("Conductivity Proxy (uS/cm)", "median"),
        median_nitrates=("Nitrate (mg/L) Derived", "median"),
        mean_latitude=("Latitude", "mean"),
        mean_longitude=("Longitude", "mean"),
    )
)
district_summary["at_risk_share"] = 1 - district_summary["safe_share"]

yearly_summary = (
    filtered_df.dropna(subset=["year"])
    .groupby("year", as_index=False)
    .agg(
        samples=("safe_flag", "size"),
        safe_share=("safe_flag", "mean"),
        avg_conductivity=("Conductivity Proxy (uS/cm)", "mean"),
        avg_nitrates=("Nitrate (mg/L) Derived", "mean"),
    )
)
yearly_summary["at_risk_share"] = 1 - yearly_summary["safe_share"]

feature_summary = (
    filtered_df.groupby("risk_label")[[
        "Potential of Hydrogen (pH)",
        "Conductivity Proxy (uS/cm)",
        "Nitrate (mg/L) Derived",
    ]]
    .median()
    .reset_index()
)

overview_col, geo_col = st.columns([1.3, 1])

with overview_col:
    st.subheader("Risk Trend", anchor=False)
    trend_chart = (
        alt.Chart(yearly_summary)
        .mark_line(point=True, strokeWidth=3, color=PRIMARY)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("at_risk_share:Q", title="At-Risk Share", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("samples:Q", title="Samples"),
                alt.Tooltip("at_risk_share:Q", title="At-Risk Share", format=".1%"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(trend_chart, width="stretch")

with geo_col:
    st.subheader("District Risk Heatmap", anchor=False)
    heatmap = (
        alt.Chart(district_summary.sort_values("at_risk_share", ascending=False))
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("at_risk_share:Q", title="At-Risk Share", axis=alt.Axis(format="%")),
            y=alt.Y("District:N", sort="-x", title="District"),
            color=alt.Color(
                "at_risk_share:Q",
                scale=alt.Scale(domain=[0, 1], range=["#8ecae6", ACCENT]),
                legend=None,
            ),
            tooltip=[
                "District",
                alt.Tooltip("samples:Q", title="Samples"),
                alt.Tooltip("at_risk_share:Q", title="At-Risk Share", format=".1%"),
                alt.Tooltip("median_ph:Q", title="Median pH", format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(heatmap, width="stretch")

middle_col, right_col = st.columns([1.2, 1])

with middle_col:
    st.subheader("Proxy Feature Space", anchor=False)
    scatter = (
        alt.Chart(filtered_df)
        .mark_circle(opacity=0.72, stroke="white", strokeWidth=0.4)
        .encode(
            x=alt.X("Potential of Hydrogen (pH):Q", title="pH"),
            y=alt.Y("Conductivity Proxy (uS/cm):Q", title="Conductivity Proxy (uS/cm)"),
            size=alt.Size("Nitrate (mg/L) Derived:Q", title="Nitrates", scale=alt.Scale(range=[45, 500])),
            color=alt.Color(
                "risk_label:N",
                title="Risk Label",
                scale=alt.Scale(domain=["Safe", "At Risk"], range=[PRIMARY, ACCENT]),
            ),
            tooltip=[
                "District",
                alt.Tooltip("year:Q", title="Year"),
                alt.Tooltip("Potential of Hydrogen (pH):Q", title="pH", format=".2f"),
                alt.Tooltip("Conductivity Proxy (uS/cm):Q", title="Conductivity", format=".1f"),
                alt.Tooltip("Nitrate (mg/L) Derived:Q", title="Nitrates", format=".2f"),
                "risk_label",
            ],
        )
        .properties(height=360)
        .interactive()
    )
    st.altair_chart(scatter, width="stretch")

with right_col:
    st.subheader("Model", anchor=False)
    model_metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Holdout Accuracy",
                "Balanced Accuracy",
                "Holdout ROC AUC",
                "3-Fold CV Accuracy",
                "3-Fold CV ROC AUC",
            ],
            "Value": [
                model_metrics["holdout_accuracy"],
                model_metrics["holdout_balanced_accuracy"],
                model_metrics["holdout_auc"],
                model_metrics["cv_accuracy_mean"],
                model_metrics["cv_auc_mean"],
            ],
        }
    )
    st.dataframe(
        model_metrics_df.style.format({"Value": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Baseline model: logistic regression on pH, conductivity proxy, and derived nitrates with median imputation, scaling, stratified split, and 3-fold CV."
    )

    st.subheader("Risk Profile Median View", anchor=False)
    feature_chart = (
        alt.Chart(feature_summary.melt("risk_label", var_name="feature", value_name="median_value"))
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("feature:N", title=None),
            y=alt.Y("median_value:Q", title="Median Value"),
            color=alt.Color(
                "risk_label:N",
                scale=alt.Scale(domain=["Safe", "At Risk"], range=[PRIMARY, ACCENT]),
                title="Risk Label",
            ),
            xOffset="risk_label:N",
            tooltip=["risk_label", "feature", alt.Tooltip("median_value:Q", format=".2f")],
        )
        .properties(height=250)
    )
    st.altair_chart(feature_chart, width="stretch")

left_tab, model_tab, data_tab = st.tabs(["Lite Diagnosis", "Model Context", "Data Explorer"])

with left_tab:
    st.subheader("Interactive Lite Diagnosis", anchor=False)
    input_cols = st.columns(3)
    with input_cols[0]:
        ph_value = st.slider("pH", min_value=4.0, max_value=10.0, value=7.4, step=0.1)
    with input_cols[1]:
        conductivity_value = st.slider(
            "Conductivity Proxy (uS/cm)",
            min_value=50,
            max_value=2500,
            value=820,
            step=10,
        )
    with input_cols[2]:
        nitrate_value = st.slider(
            "Nitrates (mg/L)",
            min_value=0.0,
            max_value=100.0,
            value=18.0,
            step=0.5,
        )

    safe_probability, ml_label, rule_label = score_sample(
        model,
        ph=ph_value,
        conductivity=conductivity_value,
        nitrates=nitrate_value,
    )

    diagnosis_cols = st.columns(3)
    with diagnosis_cols[0]:
        st.metric("ML Safe Probability", f"{safe_probability * 100:.1f}%")
    with diagnosis_cols[1]:
        st.metric("ML Prediction", ml_label)
    with diagnosis_cols[2]:
        st.metric("Rule-Based Verdict", rule_label)

    st.progress(min(max(safe_probability, 0.0), 1.0))
    st.caption(
        "This panel is useful for school demos, low-cost field trials, or quick what-if scenarios using the lite 3-feature model."
    )

with model_tab:
    st.subheader("What This MVP Is Doing", anchor=False)
    st.markdown(
        """
        - Uses the raw Andhra Pradesh water-quality dataset already present in this workspace.
        - Derives the lite inputs from existing columns:
          pH from `Potential of Hydrogen (pH)`, conductivity proxy from `Total Dissolved Solids / 0.64`, and nitrates from `Nitrate N * 4.43`.
        - Creates a simple proxy target:
          `Safe` when `6.5 <= pH <= 8.5`, conductivity proxy `<= 1500`, and nitrates `<= 45`.
        - Trains a baseline logistic-regression classifier as an MVP benchmark.
        """
    )

    source_cols = st.columns(3)
    with source_cols[0]:
        st.metric("Raw Rows", f"{len(raw_df):,}")
    with source_cols[1]:
        st.metric("Lite Rows", f"{len(lite_df):,}")
    with source_cols[2]:
        st.metric("Districts", f"{lite_df['District'].nunique()}")


with data_tab:
    st.subheader("Filtered Dataset", anchor=False)
    explorer_cols = st.columns(2)
    with explorer_cols[0]:
        st.dataframe(
            district_summary.sort_values("at_risk_share", ascending=False),
            width="stretch",
            hide_index=True,
        )
    with explorer_cols[1]:
        st.dataframe(
            filtered_df.sort_values("timestamp", ascending=False).head(200),
            width="stretch",
            hide_index=True,
        )
