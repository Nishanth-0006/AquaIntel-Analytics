# 🌊 AquaIntel Analytics — Water Quality Intelligence Dashboard

AquaIntel Analytics is a **data-driven water quality monitoring and risk analysis platform** built using Streamlit. It enables users to analyze, visualize, and predict water safety using environmental parameters and machine learning.

---

## 🚀 Project Overview

This application is designed to provide **actionable insights on water quality** using real-world datasets. It combines:

* 📊 Data Analytics
* 🤖 Machine Learning
* 🗺️ GIS-Based Visualization

to help identify whether water is **safe for consumption** and highlight **risk-prone regions**.

---

## 🎯 Core Functionalities

### 📊 Exploratory Data Analysis

* Analyze water quality parameters across regions
* Distribution plots, trends, and correlations
* Data-driven insights for decision-making

---

### 🤖 Predictive Modeling

* Built using:

  * Random Forest
  * XGBoost
  * Hybrid ML models
* Predicts water safety classification:

  * Safe
  * Unsafe

---

### 🗺️ GIS Water Quality Mapping

* Interactive visualization of water quality across regions
* Multiple map styles (terrain, dark, satellite, etc.)
* Marker-based and heatmap representations

---

## 🌊 Upload & Risk Mapping (Enhanced Feature)

The application includes an advanced **Upload Tab** that allows users to:

### 📥 Upload Custom Data

* Supports:

  * CSV files
  * Excel files

### 📌 Required Inputs

* District / Location
* pH
* Conductivity
* Nitrate

---

### ⚙️ Automated Risk Analysis

Once uploaded, the system:

* Cleans and processes the dataset
* Evaluates water quality using key parameters
* Generates a **Risk Level Classification**:

| Risk Level  | Description                       |
| ----------- | --------------------------------- |
| ✅ Safe      | All parameters within safe limits |
| ⚠️ Moderate | One parameter exceeds safe limit  |
| ❌ Unsafe    | Multiple parameters exceed limits |

---

### 🗺️ District-Level Risk Map

* Displays **interactive map visualization**
* Color-coded risk levels:

  * 🟢 Safe
  * 🟡 Moderate
  * 🔴 Unsafe
* Hover insights include:

  * District name
  * pH value
  * Conductivity
  * Nitrate level
  * Risk classification

---

## 🎨 User Interface Design

The dashboard follows a **modern water-themed UI**:

* 🌊 Gradient blue sidebar (ocean-inspired)
* 💧 Clean card-based layout
* 🎯 High readability with optimized text contrast
* 📱 Responsive and interactive components

---

## 🛠️ Tech Stack

| Category         | Tools Used            |
| ---------------- | --------------------- |
| Frontend         | Streamlit             |
| Data Processing  | Pandas, NumPy         |
| Visualization    | Plotly, Mapbox        |
| Machine Learning | Scikit-learn, XGBoost |
| Deployment       | Streamlit / Local     |

---

## 📂 Project Structure

```
AquaIntel-Analytics/
│── app.py
│── data/
│   └── raw/
│── utils/
│── models/
│── notebooks/
│── requirements.txt
│── README.md
```
---
## 🔄 Workflow

```text
1. Data Input
   ↓
   Upload dataset (CSV/Excel) or use existing data

2. Data Preprocessing
   ↓
   Clean data, handle missing values, select required parameters

3. Feature Extraction
   ↓
   Extract pH, Conductivity, Nitrate

4. Risk Analysis
   ↓
   Apply threshold logic to classify:
   Safe / Moderate / Unsafe

5. Prediction (Optional)
   ↓
   Use ML models (Random Forest, XGBoost)

6. Visualization
   ↓
   Display charts, insights, and GIS-based risk map

7. Output
   ↓
   Show water quality status and district-wise risk levels
```
---


## 💡 Tips for Best Screenshots

* Use **dark + water theme UI (your updated design)**
* Show:

  * Filters in sidebar
  * Map with colored markers
  * Metrics/cards visible
* Avoid clutter — keep it clean and focused

---

## ▶️ Getting Started

```bash
# Clone the repository
git clone https://github.com/NitishRH07/AquaIntel-Analytics.git

# Navigate into the project
cd AquaIntel-Analytics

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 📈 Applications

* Water resource monitoring
* Environmental analytics
* Government decision support systems
* Smart city infrastructure
* Academic and research projects

---

## ⚠️ Limitations

* Requires structured dataset format
* Mapping requires location coordinates
* Model performance depends on training data quality

---

## 🔮 Future Scope

* Real-time sensor data integration (IoT)
* Advanced deep learning models
* Automated geo-mapping for districts
* Cloud-based analytics and storage
* Mobile-friendly UI enhancements

---

## 👨‍💻 Author

Cohort 15 DSML

---

## 📜 License

This project is intended for **academic, research, and learning purposes**.
