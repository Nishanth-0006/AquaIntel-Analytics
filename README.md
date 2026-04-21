# 💧 AquaIntel Analytics
**Intelligence-First Water Quality Monitoring & Risk Assessment**
*DSML Cohort 15 | IIMSTC | SDG 6 Aligned*

---

## Project Structure
```
aquaintel/
├── app.py                    # Streamlit dashboard (run this)
├── requirements.txt
├── data/                     # ← DROP YOUR CWC CSVs HERE
│   └── raw/                  # Optional sub-folder; auto-discovered
├── models/                   # Trained models (auto-generated)
├── figures/                  # EDA & model plots (auto-generated)
├── utils/
│   └── data_loader.py        # Data loading, WQI, preprocessing
└── notebooks/
    ├── eda.py                # EDA script → saves all figures
    └── model_dev.py          # Model training script
```

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Drop your CSV files into /data/
#    (CWC SWQ format — any state, any year)
cp swq_manual_*.csv data/

# 2. Run EDA
python notebooks/eda.py          # generates /figures/*.png

# 3. Train models
python notebooks/model_dev.py    # generates /models/*.pkl

# 4. Launch dashboard
streamlit run app.py
```

If no CSVs are found, the system auto-generates **synthetic demo data** so you can explore the full UI immediately.

---

## Adding New Datasets

Just drop new CSVs into `/data/`. The loader auto-discovers all `*.csv` files recursively and maps column names via the alias table in `data_loader.py`. No code changes needed.

To add a new state column alias, add one line to `RENAME_MAP` in `utils/data_loader.py`.

---

## Models

| Model       | Features                  | Target Accuracy | Use Case                    |
|-------------|---------------------------|----------------|-----------------------------|
| RF Full     | All available parameters  | ≥ 92%          | Municipalities, industry    |
| XGB Full    | All available parameters  | ≥ 93%          | Municipalities, industry    |

All models use:
- Rank Centroid Weighting for WQI computation
- BIS 10500:2012 standards for labelling
- StratifiedKFold cross-validation (k=5)
- SMOTE for class imbalance handling

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 📊 Overview | KPIs, WQI distribution, BIS exceedance, state comparison |
| 🗺️ Risk Heatmap | Spatial density + scatter map by quality |
| 📈 Trends & Seasonal | Year-over-year trends, monsoon vs non-monsoon |
| 🔬 Parameter Analysis | Deep dive into any parameter, correlations |
| 🤖 Predict (ML) | Real-time prediction with gauge + compliance check |
| 📤 Upload New Data | Drag & drop new CSVs for instant analysis |

---

## Data Sources
- **CWC SWQ Manual Chemical Parameters** (1961–2020)
- States: Andhra Pradesh, Jharkhand, Karnataka, Kerala, Maharashtra, Meghalaya, Manipur, Mizoram
- Standards: BIS 10500:2012 (Indian Drinking Water Standards)

## Phase 2 Roadmap
- Real-time IoT sensor integration
- NWDP / GEMStat API connection
- SHAP explainability panel
- Government portal embed
