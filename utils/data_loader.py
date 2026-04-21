"""
AquaIntel Analytics — Data Loader
Handles CWC SWQ manual chemical parameter CSVs.
Flexible: drop new state CSVs into /data/raw/ and they are auto-discovered.
"""

import os
import glob
import pandas as pd
import numpy as np

# ── Standard CWC column aliases ──────────────────────────────────────────────
# CWC files have inconsistent header casing / spacing across states.
# We normalise everything to lowercase-underscore.
RENAME_MAP = {
    # Station / location
    "station name": "station_name",
    "station_name": "station_name",
    "stationname": "station_name",
    "location": "station_name",
    "site name": "station_name",

    "state": "state",
    "state name": "state",

    "river": "river_name",
    "river name": "river_name",
    "rivername": "river_name",
    "basin": "basin",
    "basin name": "basin",

    "year": "year",
    "month": "month",

    "latitude": "latitude",
    "lat": "latitude",
    "longitude": "longitude",
    "lon": "longitude",
    "long": "longitude",

    # Core parameters
    "ph": "pH",
    "p h": "pH",

    "conductivity": "conductivity",
    "cond": "conductivity",
    "ec": "conductivity",
    "electrical conductivity": "conductivity",

    "turbidity": "turbidity",
    "turb": "turbidity",

    "temperature": "temperature",
    "temp": "temperature",
    "water temperature": "temperature",

    "dissolved oxygen": "dissolved_oxygen",
    "do": "dissolved_oxygen",
    "d.o.": "dissolved_oxygen",

    "bod": "BOD",
    "b.o.d.": "BOD",
    "biochemical oxygen demand": "BOD",

    "cod": "COD",
    "chemical oxygen demand": "COD",

    "total dissolved solids": "TDS",
    "tds": "TDS",

    "total suspended solids": "TSS",
    "tss": "TSS",

    "nitrate": "nitrates",
    "nitrates": "nitrates",
    "no3": "nitrates",
    "no3-n": "nitrates",

    "nitrite": "nitrites",
    "no2": "nitrites",

    "ammonia": "ammonia",
    "nh3": "ammonia",
    "nh4": "ammonia",
    "ammonical nitrogen": "ammonia",

    "phosphate": "phosphate",
    "total phosphorus": "phosphate",
    "po4": "phosphate",

    "chloride": "chloride",
    "cl": "chloride",

    "fluoride": "fluoride",
    "f": "fluoride",

    "sulphate": "sulphate",
    "sulfate": "sulphate",
    "so4": "sulphate",

    "total hardness": "total_hardness",
    "hardness": "total_hardness",

    "calcium": "calcium",
    "ca": "calcium",

    "magnesium": "magnesium",
    "mg": "magnesium",

    "sodium": "sodium",
    "na": "sodium",

    "potassium": "potassium",
    "k": "potassium",

    "total coliform": "total_coliform",
    "coliform": "total_coliform",
    "fecal coliform": "fecal_coliform",
    "e. coli": "fecal_coliform",

    "arsenic": "arsenic",
    "as": "arsenic",

    "lead": "lead",
    "pb": "lead",

    "iron": "iron",
    "fe": "iron",

    "manganese": "manganese",
    "mn": "manganese",

    "zinc": "zinc",
    "zn": "zinc",

    "copper": "copper",
    "cu": "copper",

    "chromium": "chromium",
    "cr": "chromium",

    "nickel": "nickel",
    "ni": "nickel",

    "cadmium": "cadmium",
    "cd": "cadmium",

    "mercury": "mercury",
    "hg": "mercury",

    "dissolved oxygen (mg/l)": "dissolved_oxygen",
    "total dissolved solids (mg/l)": "TDS",
    "potential of hydrogen (ph)": "pH",
    "nitrate (mg/l)": "nitrates",
    "amonia n (mgn/l)": "ammonia",
    "fluoride (mg/l)": "fluoride",
    "sulphate (mg/l)": "sulphate",
    "chloride (mg/l)": "chloride",
    "calcium (mg/l)": "calcium",
    "magnesium (mg/l)": "magnesium",
    "sodium (mg/l)": "sodium",
    "potassium (mg/l)": "potassium",
    "arsenic (mg/l)": "arsenic",
    "cadmium (mg/l)": "cadmium",
    "chromium (mg/l)": "chromium",
    "copper (mg/l)": "copper",
    "iron(mg/l)": "iron",
    "lead (mg/l)": "lead",
    "manganese (mg/l)": "manganese",
    "mercury(mg/l)": "mercury",
    "nickel (mg/l)": "nickel",
    "zinc (mg/l)": "zinc",
    "total hardness (mgcaco3/l)": "total_hardness",
    "hardness_magnesium (mg/l as caco3)": "magnesium",
    "hardness calcium (mgcaco3/l)": "calcium",
    "total phosphorus (mgp/l)": "phosphate",
    "nitrite n+nitrate n (mgn/l)": "nitrates",
    "bicarbonate (mg/l)": "bicarbonate",
    "carbonate (mg/l)": "carbonate",
    "boron (mg/l)": "boron",
    "data acquisition time": "data acquisition time",
}

# ── BIS 10500 : 2012 drinking-water standards (for WQI labelling) ─────────
BIS_STANDARDS = {
    "pH":               {"min": 6.5, "max": 8.5},
    "conductivity":     {"max": 750},       # µS/cm
    "turbidity":        {"max": 1},         # NTU
    "dissolved_oxygen": {"min": 6},         # mg/L (higher = better)
    "BOD":              {"max": 3},         # mg/L
    "COD":              {"max": 10},
    "TDS":              {"max": 500},       # mg/L
    "nitrates":         {"max": 45},        # mg/L
    "ammonia":          {"max": 0.5},
    "phosphate":        {"max": 0.1},
    "chloride":         {"max": 250},
    "fluoride":         {"max": 1.0},
    "sulphate":         {"max": 200},
    "total_hardness":   {"max": 300},
    "arsenic":          {"max": 0.01},
    "lead":             {"max": 0.01},
    "iron":             {"max": 0.3},
    "manganese":        {"max": 0.1},
    "total_coliform":   {"max": 0},         # MPN/100mL (0 = safe)
    "fecal_coliform":   {"max": 0},
}

# All numeric feature columns (used for full model)
CORE_FEATURES = [
    "pH", "conductivity", "turbidity", "temperature",
    "dissolved_oxygen", "BOD", "COD", "TDS", "TSS",
    "nitrates", "nitrites", "ammonia", "phosphate",
    "chloride", "fluoride", "sulphate", "total_hardness",
    "calcium", "magnesium", "arsenic", "lead", "iron",
    "manganese", "total_coliform", "fecal_coliform",
]

STATE_CODES = {
    "ap": "Andhra Pradesh",
    "jh": "Jharkhand",
    "ka": "Karnataka",
    "kl": "Kerala",
    "mh": "Maharashtra",
    "ml": "Meghalaya",
    "mn": "Manipur",
    "mz": "Mizoram",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-strip column names, apply alias map, and remove duplicates."""
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    
    # Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    return df


def load_single_csv(path: str, state_code: str = None) -> pd.DataFrame:
    """Load one CWC CSV, normalise columns, tag with state."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1", low_memory=False)

    df = _normalise_columns(df)

    # Infer state from filename if not given
    if state_code is None:
        fname = os.path.basename(path).lower()
        for code in STATE_CODES:
            if f"_{code}_" in fname or f"_{code}." in fname:
                state_code = code
                break

    if state_code and "state" not in df.columns:
        df["state"] = STATE_CODES.get(state_code, state_code.upper())
    elif "state" not in df.columns:
        df["state"] = "Unknown"

    # Numeric coercion for all CORE_FEATURES present
    for col in CORE_FEATURES:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Year / month coercion
    if "data acquisition time" in df.columns:
        dt_col = df["data acquisition time"]
        if isinstance(dt_col, pd.DataFrame):
            dt_col = dt_col.iloc[:, 0]
        dt = pd.to_datetime(dt_col, errors="coerce", format="mixed", dayfirst=True)
        if "year" not in df.columns:
            df["year"] = dt.dt.year
        if "month" not in df.columns:
            df["month"] = dt.dt.month

    for col in ["year", "month"]:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_all_csvs(data_dir: str) -> pd.DataFrame:
    """
    Auto-discover all CSVs in data_dir (and sub-dirs) and concat.
    Designed to be forward-compatible: just drop new files in.
    """
    pattern = os.path.join(data_dir, "**", "*.csv")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    errors = []
    for p in paths:
        try:
            frames.append(load_single_csv(p))
        except Exception as e:
            errors.append(f"  - {os.path.basename(p)}: {e}")
            print(f"[WARN] Could not load {p}: {e}")

    if not frames:
        error_msg = f"No CSV files could be loaded from {data_dir}. Errors:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    if errors:
        print(f"[INFO] Loaded {len(frames)}/{len(paths)} CSV files. {len(errors)} failed.")

    df = pd.concat(frames, ignore_index=True)
    return df


def compute_wqi(df: pd.DataFrame) -> pd.Series:
    """
    Rank-Centroid–Weighted WQI (0–100; higher = worse).
    Uses available parameters vs BIS standards.
    Returns a Series of float WQI values.
    """
    available = [p for p in BIS_STANDARDS if p in df.columns]
    n = len(available)
    if n == 0:
        return pd.Series(np.nan, index=df.index)

    # Rank centroid weights: w_i = (1/rank_i) / sum(1/rank_j)
    ranks = {p: i + 1 for i, p in enumerate(available)}  # rank 1 = most important
    raw_weights = {p: 1 / ranks[p] for p in available}
    total = sum(raw_weights.values())
    weights = {p: raw_weights[p] / total for p in available}

    scores = pd.DataFrame(index=df.index)
    for param, std in BIS_STANDARDS.items():
        if param not in df.columns:
            continue
        val = df[param].astype(float)
        if param == "dissolved_oxygen":
            # Higher DO is better; sub-standard = val < min
            ideal = std.get("min", 6)
            qi = ((ideal - val) / ideal * 100).clip(0, 100)
        elif "max" in std:
            qi = (val / std["max"] * 100).clip(0, 100)
        else:
            # pH: penalty from ideal midpoint
            mid = (std["min"] + std["max"]) / 2
            qi = (abs(val - mid) / ((std["max"] - std["min"]) / 2) * 100).clip(0, 100)
        scores[param] = qi * weights[param]

    wqi = scores.sum(axis=1)
    return wqi


def label_water_quality(wqi: pd.Series) -> pd.Series:
    """Convert WQI to human-readable category."""
    bins   = [0,   25,  50,  75,  100]
    labels = ["Excellent", "Good", "Poor", "Very Poor"]
    return pd.cut(wqi, bins=bins, labels=labels, include_lowest=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = df.copy()

    # Drop columns that are entirely NaN
    df.dropna(axis=1, how="all", inplace=True)

    # Compute WQI
    df["WQI"] = compute_wqi(df)
    df["water_quality"] = label_water_quality(df["WQI"])

    # Binary safe/unsafe label for classification
    df["is_safe"] = (df["WQI"] <= 50).astype(int)

    # Drop rows with no WQI (all parameters missing)
    df = df.dropna(subset=["WQI"])

    return df


def generate_synthetic_cwc(n=5000, seed=42) -> pd.DataFrame:
    """
    Generate realistic synthetic CWC-style data for demo / testing.
    Used when actual files are not yet available in the container.
    """
    rng = np.random.default_rng(seed)
    states = list(STATE_CODES.values())
    months = list(range(1, 13))
    years  = list(range(2000, 2021))

    rivers_by_state = {
        "Andhra Pradesh":  ["Krishna", "Godavari", "Tungabhadra"],
        "Jharkhand":       ["Damodar", "Subarnarekha", "Brahmani"],
        "Karnataka":       ["Cauvery", "Krishna", "Tungabhadra", "Sharavathi"],
        "Kerala":          ["Periyar", "Pamba", "Chaliyar"],
        "Maharashtra":     ["Godavari", "Krishna", "Tapi", "Wardha"],
        "Meghalaya":       ["Umiam", "Myntdu", "Digaru"],
        "Manipur":         ["Barak", "Iril", "Thoubal"],
        "Mizoram":         ["Tlawng", "Tuirial", "Chhimtuipui"],
    }

    rows = []
    for _ in range(n):
        state = rng.choice(states)
        river = rng.choice(rivers_by_state[state])
        year  = int(rng.choice(years))
        month = int(rng.choice(months))

        # Seasonal variation
        monsoon = month in [6, 7, 8, 9]
        pollution_factor = rng.uniform(0.5, 2.0)

        row = {
            "state":        state,
            "river_name":   river,
            "year":         year,
            "month":        month,
            "station_name": f"{river}-ST{rng.integers(1,20):02d}",
            "latitude":     float(rng.uniform(8, 30)),
            "longitude":    float(rng.uniform(72, 97)),

            "pH":               float(np.clip(rng.normal(7.2, 0.6 * pollution_factor), 4, 10)),
            "conductivity":     float(np.clip(rng.lognormal(5.5, 0.8) * pollution_factor, 50, 3000)),
            "turbidity":        float(np.clip(rng.lognormal(0.5, 1.0) * (2 if monsoon else 1), 0.1, 100)),
            "temperature":      float(np.clip(rng.normal(26 + (2 if monsoon else -2), 3), 10, 40)),
            "dissolved_oxygen": float(np.clip(rng.normal(7.5 / pollution_factor, 1.5), 1, 14)),
            "BOD":              float(np.clip(rng.lognormal(0.8, 0.9) * pollution_factor, 0.1, 50)),
            "COD":              float(np.clip(rng.lognormal(1.5, 0.9) * pollution_factor, 0.5, 200)),
            "TDS":              float(np.clip(rng.lognormal(5.0, 0.9) * pollution_factor, 50, 3000)),
            "nitrates":         float(np.clip(rng.lognormal(1.5, 1.1) * pollution_factor, 0, 120)),
            "ammonia":          float(np.clip(rng.lognormal(-0.5, 1.2) * pollution_factor, 0, 10)),
            "phosphate":        float(np.clip(rng.lognormal(-2, 1.3) * pollution_factor, 0, 5)),
            "chloride":         float(np.clip(rng.lognormal(3.5, 1.0) * pollution_factor, 5, 1000)),
            "fluoride":         float(np.clip(rng.lognormal(-0.5, 0.8) * pollution_factor, 0, 5)),
            "sulphate":         float(np.clip(rng.lognormal(3.5, 1.0) * pollution_factor, 5, 800)),
            "total_hardness":   float(np.clip(rng.lognormal(4.5, 0.8) * pollution_factor, 20, 1500)),
            "iron":             float(np.clip(rng.lognormal(-1.5, 1.5) * pollution_factor, 0, 5)),
            "arsenic":          float(np.clip(rng.lognormal(-5, 1.5) * pollution_factor, 0, 0.1)),
            "total_coliform":   float(np.clip(rng.lognormal(3, 2) * pollution_factor, 0, 50000)),
        }
        rows.append(row)

    return pd.DataFrame(rows)
