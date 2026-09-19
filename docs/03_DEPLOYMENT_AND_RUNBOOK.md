# Financial Forecasting Frontier: Deployment & Operational Runbook
## System Execution, Distributed Pipeline Operations & Model Serving

This runbook provides end-to-end instructions for deploying, executing, and maintaining the **Financial Forecasting Frontier: Distributed Machine Learning for Banking Analytics** platform.

---

## 1. System Requirements & Prerequisites

### 1.1 Hardware Specifications
- **CPU:** Quad-Core Processor (x86_64, minimum 4 cores recommended for parallelism benchmarks).
- **RAM:** Minimum 8 GB (16 GB recommended for in-memory distributed datasets and SMOTE oversampling).
- **Disk Space:** 5 GB free disk space for dataset cache, model binaries, and serialized pipelines.
- **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 20.04+).

### 1.2 Software Dependencies
- **Python Runtime:** Python 3.11 (configured via `py -3.11` or virtual environment).
- **Core ML & Data Packages:** `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `xgboost>=2.0.0`, `imbalanced-learn>=0.11.0`, `shap>=0.42.0`, `joblib>=1.3.0`.
- **Distributed Computing & Visualization:** `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `streamlit>=1.28.0`, `nbformat>=5.9.0`, `nbclient>=0.8.0`, `ipykernel>=6.25.0`.

---

## 2. Environment Setup & Dependency Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/KARTHIKEYAN-31/Financial-Forecasting-Frontier-Distributed-ML.git
cd "Financial Forecasting Frontier Distributed ML"
```

### Step 2: Initialize Virtual Environment (Recommended)
```bash
# Create Python 3.11 Virtual Environment
py -3.11 -m venv .venv

# Activate Environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### Step 3: Install Package Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Running Distributed Big Data Pipelines & Scripts

The `distributed_system/` module contains standalone production modules simulating enterprise Hadoop, Hive, and Spark workflows:

### 3.1 Run Apache Spark Distributed EDA & MLlib Pipeline
Executes distributed SQL queries on in-memory views, performs VectorAssembler feature encoding, and trains distributed Random Forest and GBTClassifiers:
```bash
python distributed_system/spark_eda_mllib.py
```
*Expected Output:*
- Prints occupational conversion breakdown table.
- Prints joint debt burden liquidity matrix.
- Outputs Spark MLlib Random Forest (ROC-AUC ~0.895) and GBTClassifier (ROC-AUC ~0.893) training logs.

### 3.2 Run Real-Time Spark Structured Streaming Simulation
Simulates micro-batch event streams, computing tumbling window totals, 20-transaction sliding averages, and automated fraud sink alerts:
```bash
python distributed_system/spark_streaming_realtime.py
```
*Expected Output:*
- Ingests 50 real-time transaction events.
- Displays Tumbling Window micro-batches (every 10 transactions).
- Displays Sliding Window moving statistics (every 5 transactions).
- Intercepts and logs critical fraud alerts for amounts $>€2,000$ or critical device risks.

### 3.3 Run Data Parallelism & Scalability Benchmarks
Benchmarks partition scaling across 1, 2, 4, and 8 partitions on 250,000 synthetic banking records and compares Map-Side Broadcast Join against Distributed Shuffle Join:
```bash
python distributed_system/data_parallelism_benchmark.py
```
*Expected Output:*
- Displays partition latency, speedup ratio ($2.02\times$ on 2 partitions, $2.45\times$ on 4 partitions), and throughput ($>12.3\text{M rows/sec}$).
- Confirms Broadcast Hash Join eliminates cluster network shuffle overhead.

---

## 4. Executing Capstone Submission Notebooks

All Jupyter notebooks are fully pre-executed and can be reviewed or re-run:

### 4.1 EDA Submission Notebook
- **File:** [`Financial_Forecasting_EDA_Submission.ipynb`](../Financial_Forecasting_EDA_Submission.ipynb)
- **Contents:** 167 cells adhering strictly to the Woolf submission template. Features 15 visual charts covering the UBM rule, demographic wrangling, correlation heatmaps, and pairplots with comprehensive business solutions.

### 4.2 ML Submission Notebook
- **File:** [`Financial_Forecasting_ML_Submission.ipynb`](../Financial_Forecasting_ML_Submission.ipynb)
- **Contents:** 305 cells adhering strictly to the Woolf template. Includes 3 formal hypothesis tests ($\chi^2$, Mann-Whitney U), outlier capping, SMOTE resampling, 3 tuned ML models (LR, RF, XGBoost), ROC/PR curve comparisons, SHAP explainability, and artifact persistence.

### 4.3 Distributed Big Data Notebook
- **File:** [`Distributed_Banking_BigData_Spark_Hive.ipynb`](../Distributed_Banking_BigData_Spark_Hive.ipynb)
- **Contents:** Unifies Hadoop HDFS DDL, Apache HiveQL analytical warehousing, Spark SQL, Spark MLlib pipelines, real-time streaming, and data parallelism plots.

---

## 5. Launching the Interactive Streamlit Web Application

The project includes an executive multi-tab web application for real-time customer propensity scoring and live stream monitoring:

```bash
streamlit run app.py
```

### Application Features:
- **Tab 1: Executive Portfolio Dashboard:** Real-time KPI cards, conversion by profession, monthly seasonality trends, and joint debt liability matrix.
- **Tab 2: Real-Time Propensity Scorer:** Interactive client parameter inputs (age, balance, loans, call duration, campaign recency). Loads `best_banking_model.joblib` to calculate instant subscription probability, assigns risk tiers, and explains key contributing factors.
- **Tab 3: Real-Time Streaming & Anomaly Detection:** Interactive live stream generator with sliding window buffers and automated fraud interception alerts.
- **Tab 4: Distributed Architecture & Big Data Benchmarks:** Interactive visualization of Hive DDL schemas, Spark pipeline diagrams, and partition scaling charts.

---

## 6. Programmatic Model Inference Guide

To integrate the trained XGBoost model into an external banking microservice:

```python
import joblib
import pandas as pd
import numpy as np

# 1. Load the serialized pipeline artifact bundle
bundle = joblib.load('best_banking_model.joblib')
model = bundle['model']
scaler = bundle['scaler']
feature_names = bundle['feature_names']
num_scale_cols = bundle['num_scale_cols']

# 2. Example Incoming Customer Record
new_customer = pd.DataFrame([{
    'age': 58,
    'job': 'retired',
    'marital': 'married',
    'education': 'tertiary',
    'default': 'no',
    'balance': 4500,
    'housing': 'no',
    'loan': 'no',
    'contact': 'cellular',
    'day': 15,
    'month': 'oct',
    'duration': 420,
    'campaign': 1,
    'pdays': 120,
    'previous': 2,
    'poutcome': 'success'
}])

# 3. Apply Feature Preprocessing
binary_map = {'no': 0, 'yes': 1}
new_customer['default'] = new_customer['default'].map(binary_map)
new_customer['housing'] = new_customer['housing'].map(binary_map)
new_customer['loan'] = new_customer['loan'].map(binary_map)
new_customer['education'] = new_customer['education'].map({'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3})

encoded_df = pd.get_dummies(new_customer, columns=['job', 'marital', 'contact', 'month', 'poutcome'], drop_first=False, dtype=int)
encoded_df['balance_to_age'] = encoded_df['balance'] / encoded_df['age']
encoded_df['was_contacted_prev'] = (encoded_df['pdays'] != -1).astype(int)
encoded_df['total_debt_count'] = encoded_df['housing'] + encoded_df['loan']
encoded_df['balance_log'] = np.sign(encoded_df['balance']) * np.log1p(np.abs(encoded_df['balance']))
encoded_df['duration_log'] = np.log1p(encoded_df['duration'])

# Align feature vector
for col in feature_names:
    if col not in encoded_df.columns:
        encoded_df[col] = 0
encoded_df = encoded_df[feature_names]
encoded_df[num_scale_cols] = scaler.transform(encoded_df[num_scale_cols])

# 4. Generate Calibrated Propensity Score
propensity_score = model.predict_proba(encoded_df)[0, 1]
print(f"Customer Term Deposit Propensity Score: {propensity_score * 100:.2f}%")
# Output: >85% (High Priority Lead)
```

---

## 7. Troubleshooting & Operational FAQs

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **Streamlit Port Conflict (Port 8501 in use)** | Another service or orphaned process is bound to port 8501. | Launch on alternate port: `streamlit run app.py --server.port 8502` |
| **XGBoost Windows loky multiprocessing warning** | Windows process spawn behavior with loky in joblib. | Safe warning; setting `n_jobs=1` or using standard Python multiprocessing resolves warning. |
| **Out of Memory during GridSearchCV** | High partition size on resource-constrained systems. | Reduce `n_jobs` to `2` in `GridSearchCV` or lower SMOTE sampling ratio. |
