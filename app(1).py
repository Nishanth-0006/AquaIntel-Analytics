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
import os
import glob


st.set_page_config(
    page_title="AquaIntel Analytics Lite",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_FOLDER = "data sets"
PRIMARY = "#0b6e4f"
ACCENT = "#d62828"


@st.cache_data(show_spinner=False)
def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_and_merge_datasets(folder_path: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()
            state_name = os.path.basename(file).split('_')[4].upper()
            if 'Copy' in state_name:
                continue
            df['data_source_file'] = os.path.basename(file)
            dfs.append(df)
        except Exception as e:
            st.warning(f"Could not load {file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No valid CSV files found in the folder")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df


@st.cache_data(show_spinner=False)
def build_lite_dataset_from_merged(merged_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_df.copy()
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
def train_baseline_model_from_data(lite_df: pd.DataFrame):
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


raw_df = load_and_merge_datasets(DATA_FOLDER)
lite_df = build_lite_dataset_from_merged(raw_df)
model, model_metrics = train_baseline_model_from_data(lite_df)

available_states = sorted(lite_df["State"].dropna().unique().tolist())
available_districts = sorted(lite_df["District"].dropna().unique().tolist())
min_year = int(lite_df["year"].dropna().min())
max_year = int(lite_df["year"].dropna().max())

st.title("AquaIntel Analytics", anchor=False)

with st.sidebar:
    st.header("Dashboard Controls", anchor=False)
    selected_states = st.multiselect(
        "States",
        options=available_states,
        default=available_states,
    )
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
    lite_df["State"].isin(selected_states)
    & lite_df["District"].isin(selected_districts)
    & lite_df["risk_label"].isin(selected_risk)
    & lite_df["year"].between(year_range[0], year_range[1], inclusive="both")
].copy()

total_filtered = len(filtered_df)
safe_share = filtered_df["safe_flag"].mean() * 100 if total_filtered else 0.0
at_risk_share = 100 - safe_share if total_filtered else 0.0
district_count = filtered_df["District"].nunique() if total_filtered else 0
state_count = filtered_df["State"].nunique() if total_filtered else 0
year_span_text = f"{year_range[0]}-{year_range[1]}"

metrics_row = st.columns(5)
with metrics_row[0]:
    st.metric("Filtered Samples", f"{total_filtered:,}")
with metrics_row[1]:
    st.metric("Safe Share", f"{safe_share:.1f}%")
with metrics_row[2]:
    st.metric("At-Risk Share", f"{at_risk_share:.1f}%")
with metrics_row[3]:
    st.metric("States", f"{state_count}")
with metrics_row[4]:
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
    st.altair_chart(trend_chart, use_container_width=True)

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
    st.altair_chart(heatmap, use_container_width=True)

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
    st.altair_chart(scatter, use_container_width=True)

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
        use_container_width=True,
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
    st.altair_chart(feature_chart, use_container_width=True)

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
    st.subheader("What This Enhanced Analytics Is Doing", anchor=False)
    st.markdown(
        """
        - Merges water-quality datasets from multiple states (AP, JH, KA, KL, MH, ML, MN, MZ) 
        - Derives the lite inputs from existing columns:
          pH from `Potential of Hydrogen (pH)`, conductivity proxy from `Total Dissolved Solids / 0.64`, and nitrates from `Nitrate N * 4.43`.
        - Creates a simple proxy target:
          `Safe` when `6.5 <= pH <= 8.5`, conductivity proxy `<= 1500`, and nitrates `<= 45`.
        - Trains a baseline logistic-regression classifier on the merged multi-state dataset.
        - Provides comprehensive EDA with state-level comparisons and data quality analysis.
        """
    )

    source_cols = st.columns(4)
    with source_cols[0]:
        st.metric("Raw Rows", f"{len(raw_df):,}")
    with source_cols[1]:
        st.metric("Lite Rows", f"{len(lite_df):,}")
    with source_cols[2]:
        st.metric("States", f"{raw_df['State'].nunique()}")
    with source_cols[3]:
        st.metric("Districts", f"{lite_df['District'].nunique()}")


with data_tab:
    st.subheader("Comprehensive EDA - Merged Dataset Analysis", anchor=False)
    
    # State-level summary
    st.subheader("State-wise Overview", anchor=False)
    state_summary = (
        filtered_df.groupby("State", as_index=False)
        .agg(
            samples=("safe_flag", "size"),
            safe_share=("safe_flag", "mean"),
            districts=("District", "nunique"),
            avg_ph=("Potential of Hydrogen (pH)", "mean"),
            avg_conductivity=("Conductivity Proxy (uS/cm)", "mean"),
            avg_nitrates=("Nitrate (mg/L) Derived", "mean"),
            year_span=("year", lambda x: f"{int(x.min())}-{int(x.max())}")
        )
    )
    state_summary["at_risk_share"] = 1 - state_summary["safe_share"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            state_summary.sort_values("samples", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    
    with col2:
        # State comparison chart
        state_chart = (
            alt.Chart(state_summary)
            .mark_bar(cornerRadiusEnd=6)
            .encode(
                x=alt.X("at_risk_share:Q", title="At-Risk Share", axis=alt.Axis(format="%")),
                y=alt.Y("State:N", sort="-x", title="State"),
                color=alt.Color(
                    "at_risk_share:Q",
                    scale=alt.Scale(domain=[0, 1], range=["#8ecae6", ACCENT]),
                    legend=None,
                ),
                tooltip=[
                    "State",
                    alt.Tooltip("samples:Q", title="Samples"),
                    alt.Tooltip("districts:Q", title="Districts"),
                    alt.Tooltip("at_risk_share:Q", title="At-Risk Share", format=".1%"),
                    alt.Tooltip("avg_ph:Q", title="Avg pH", format=".2f"),
                ],
            )
            .properties(height=400)
        )
        st.altair_chart(state_chart, use_container_width=True)
    
    # Data quality metrics
    st.subheader("Data Quality & Completeness", anchor=False)
    
    # Missing values analysis
    missing_analysis = []
    for col in ["Potential of Hydrogen (pH)", "Total Dissolved Solids (mg/L)", "Nitrate N (mgN/L)"]:
        missing_count = raw_df[col].isna().sum()
        missing_pct = (missing_count / len(raw_df)) * 100
        missing_analysis.append({
            "Parameter": col,
            "Missing Values": missing_count,
            "Missing %": f"{missing_pct:.2f}%",
            "Available Values": len(raw_df) - missing_count
        })
    
    missing_df = pd.DataFrame(missing_analysis)
    col3, col4 = st.columns(2)
    
    with col3:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    with col4:
        # Temporal coverage by state
        temporal_coverage = (
            filtered_df.groupby(["State", "year"], as_index=False)
            .size()
            .rename(columns={"size": "samples"})
        )
        
        temporal_chart = (
            alt.Chart(temporal_coverage)
            .mark_line(opacity=0.8, strokeWidth=2)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("samples:Q", title="Sample Count"),
                color=alt.Color("State:N", title="State"),
                tooltip=["State", "year", "samples"]
            )
            .properties(height=300)
        )
        st.altair_chart(temporal_chart, use_container_width=True)
    
    # Feature distribution analysis
    st.subheader("Feature Distribution Analysis", anchor=False)
    
    col5, col6 = st.columns(2)
    
    with col5:
        # pH distribution by risk
        ph_hist = (
            alt.Chart(filtered_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("Potential of Hydrogen (pH):Q", bin=alt.Bin(maxbins=30), title="pH"),
                y=alt.Y("count()", title="Count"),
                color=alt.Color(
                    "risk_label:N",
                    scale=alt.Scale(domain=["Safe", "At Risk"], range=[PRIMARY, ACCENT]),
                    title="Risk Label"
                )
            )
            .properties(height=250)
        )
        st.altair_chart(ph_hist, use_container_width=True)
    
    with col6:
        # Conductivity distribution by risk
        cond_hist = (
            alt.Chart(filtered_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("Conductivity Proxy (uS/cm):Q", bin=alt.Bin(maxbins=30), title="Conductivity Proxy (uS/cm)"),
                y=alt.Y("count()", title="Count"),
                color=alt.Color(
                    "risk_label:N",
                    scale=alt.Scale(domain=["Safe", "At Risk"], range=[PRIMARY, ACCENT]),
                    title="Risk Label"
                )
            )
            .properties(height=250)
        )
        st.altair_chart(cond_hist, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations", anchor=False)
    
    correlation_data = filtered_df[[
        "Potential of Hydrogen (pH)",
        "Conductivity Proxy (uS/cm)", 
        "Nitrate (mg/L) Derived",
        "safe_flag"
    ]].corr()
    
    # Convert correlation matrix to long format for Altair
    corr_long = correlation_data.reset_index().melt('index')
    corr_long.columns = ['var1', 'var2', 'correlation']
    
    corr_heatmap = (
        alt.Chart(corr_long)
        .mark_rect()
        .encode(
            x='var1:N',
            y='var2:N',
            color=alt.Color('correlation:Q', 
                          scale=alt.Scale(domain=[-1, 1], scheme='redblue')),
            tooltip=['var1', 'var2', alt.Tooltip('correlation:Q', format='.3f')]
        )
        .properties(width=400, height=300)
    )
    
    col7, col8 = st.columns([1, 1])
    with col7:
        st.altair_chart(corr_heatmap, use_container_width=True)
    
    with col8:
        st.dataframe(correlation_data.round(3), use_container_width=True)
    
    # Detailed data explorer
    st.subheader("Filtered Dataset Explorer", anchor=False)
    explorer_cols = st.columns(2)
    with explorer_cols[0]:
        st.dataframe(
            district_summary.sort_values("at_risk_share", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    with explorer_cols[1]:
        st.dataframe(
            filtered_df.sort_values("timestamp", ascending=False).head(200),
            use_container_width=True,
            hide_index=True,
        )
