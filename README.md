# Financial Forecasting Frontier: Distributed Machine Learning for Banking Analytics

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Apache_Spark-3.5-orange.svg)](https://spark.apache.org/)
[![Hive](https://img.shields.io/badge/Apache_Hive-3.1-yellow.svg)](https://hive.apache.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Submission-Ready-brightgreen.svg)]()

> **An Enterprise Distributed Machine Learning, Real-Time Streaming, and Big Data Analytical Architecture for Banking Term Deposit Propensity Forecasting & Campaign Optimization.**

---

## Project Overview

In the modern commercial banking sector, direct marketing campaigns represent a major revenue driver for deposit mobilization and liquidity management. However, traditional untargeted outreach yields low conversion rates (under 12%), inflates customer acquisition costs, exhausts call center resources, and damages brand perception through communication fatigue.

This capstone project delivers a comprehensive, submission-ready machine learning and big data analytics platform utilizing the Portuguese retail banking marketing dataset (4,521 customer records $\times$ 17 features). The project combines:
1. **Hadoop HDFS Storage & Apache Hive Warehousing** for scalable data management.
2. **Apache Spark Distributed Compute & Spark SQL** for in-memory exploratory data analysis.
3. **Distributed Spark MLlib & Multi-Model Machine Learning** (Logistic Regression, Random Forest, XGBoost) with 5-fold cross-validation and hyperparameter optimization.
4. **Spark Structured Streaming Simulation** with tumbling/sliding window operations and real-time fraud/anomaly detection alerts.
5. **Data Parallelism & Scalability Benchmarking** demonstrating throughput scaling and Map-Side Broadcast Join optimization.
6. **Model Explainability (SHAP)** delivering auditable global and local feature attributions.
7. **Interactive Streamlit Web Dashboard** for real-time customer propensity scoring and stream monitoring.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      1. DATA INGESTION & HDFS TIER                          │
│  - Raw Banking Telemetry (CSV / Streaming Events)                           │
│  - Hadoop Distributed File System (HDFS) Replicated Storage                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                2. APACHE HIVE ENTERPRISE ANALYTICAL WAREHOUSE                │
│  - External & Managed Tables stored in Apache ORC Format (Snappy)           │
│  - Partitioned by Month (`month`) & Profession (`job`)                      │
│  - Clustered by Customer Age (`age`) into 6 Buckets sorted by `balance DESC`│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│   3. BATCH DISTRIBUTED ENGINE        │  │ 4. REAL-TIME STREAMING ENGINE     │
│  - Apache Spark In-Memory Compute    │  │  - Spark Structured Streaming     │
│  - Spark SQL Distributed Views       │  │  - Micro-batch Tumbling Windows   │
│  - Spark MLlib Distributed Pipelines │  │  - Sliding 20-Txn Moving Windows  │
│  - Data Parallelism & Partitioning   │  │  - Real-Time Anomaly & Fraud Sink │
└──────────────────┬───────────────────┘  └─────────────────┬─────────────────┘
                   │                                        │
                   └───────────────────┬────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      5. PRODUCTION ML & EXPLAINABILITY                      │
│  - Preprocessing: RobustScaler + Categorical Encoding + SMOTE Resampling   │
│  - Model Suite: Logistic Regression, Random Forest, Tuned XGBoost (Champion)│
│  - Post-Hoc Explainability: SHAP (SHapley Additive exPlanations)             │
│  - Artifact Serialization: `best_banking_model.joblib`                      │
│  - Interactive Dashboard: Streamlit Application (`app.py`)                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository & Submission Deliverables Structure

```text
Financial Forecasting Frontier Distributed ML/
├── dataset/
│   └── dataset.csv                                 # Portuguese Banking Marketing Dataset (4,521 x 17)
├── docs/                                           # Project Documentation Suite & Reports
│   ├── 01_FINAL_PROJECT_REPORT.md                  # Comprehensive Technical Capstone Report
│   ├── 02_INTERVIEW_QA_PREPARATION.md              # Viva & Technical Defense Preparation Guide
│   ├── 03_DEPLOYMENT_AND_RUNBOOK.md                # System Execution & Operational Runbook
│   └── 04_EVALUATION_RUBRIC_COMPLIANCE.md          # 100% Evaluation Criteria Compliance Matrix
├── sample template/
│   ├── Sample EDA Submission Template.ipynb        # Reference EDA Template
│   └── Sample ML Submission Template.ipynb         # Reference ML Template
├── distributed_system/
│   ├── hadoop_hive_setup.hql                       # Hive DDL, Partitioning, Bucketing & 6 Analytical Queries
│   ├── spark_eda_mllib.py                          # Distributed Spark SQL & Spark MLlib Pipelines
│   ├── spark_streaming_realtime.py                 # Spark Streaming & Windowed Anomaly Detection
│   └── data_parallelism_benchmark.py               # Data Parallelism & Broadcast Join Scaling Experiments
├── pipeline_artifacts/
│   └── best_banking_model.joblib                   # Serialized Model, Scaler & Pipeline Bundle
├── Financial_Forecasting_EDA_Submission.ipynb      # Complete Executed EDA Submission Notebook (15 Charts)
├── Financial_Forecasting_ML_Submission.ipynb       # Complete Executed ML Submission Notebook (Hypothesis, SMOTE, Tuning, SHAP)
├── Distributed_Banking_BigData_Spark_Hive.ipynb    # Executed Big Data & Distributed System Notebook
├── app.py                                          # Streamlit Interactive Dashboard & Propensity Scorer
├── best_banking_model.joblib                       # Standalone Production Model Artifact
├── requirements.txt                                # Python Dependencies
└── README.md                                       # Master Project Documentation & Execution Guide
```

---

## Documentation & Submission Suite

All project submission documents are located in the [`docs/`](docs/) directory:

- [**01. Final Technical Capstone Report**](docs/01_FINAL_PROJECT_REPORT.md): In-depth system architecture, dataset relationships, model performance metrics, and distributed implementation.
- [**02. Interview Q&A Preparation Guide**](docs/02_INTERVIEW_QA_PREPARATION.md): Comprehensive answers to core follow-up questions (Hadoop/Hive, Spark SQL, MLlib, SMOTE, XGBoost vs RF, SHAP, streaming, and business ROI).
- [**03. Deployment & Operational Runbook**](docs/03_DEPLOYMENT_AND_RUNBOOK.md): Step-by-step reproduction guide for local setup, Spark distributed scripts, Streamlit execution, and programmatic API inference.
- [**04. Evaluation Rubric Compliance Matrix**](docs/04_EVALUATION_RUBRIC_COMPLIANCE.md): 100% compliance breakdown across all 8 evaluation criteria.

---

## Key Statistical Findings & Hypothesis Testing

| Hypothesis | Factor Tested | Test Applied | p-Value | Conclusion |
| :--- | :--- | :--- | :---: | :--- |
| **Hypothesis 1** | Prior Campaign Outcome (`poutcome`) vs Subscription | Chi-Square $(\chi^2)$ Test | **$2.45 \times 10^{-58}$** | Highly significant; prior success converts at **64.3%** |
| **Hypothesis 2** | Call Duration (`duration`) vs Subscription | Mann-Whitney U Test | **$2.01 \times 10^{-99}$** | Highly significant; subscribers average **552s vs 221s** |
| **Hypothesis 3** | Occupational Category (`job`) vs Subscription | Chi-Square $(\chi^2)$ Test | **$1.12 \times 10^{-11}$** | Significant; retirees (**23.5%**) & students (**22.6%**) lead |

---

## Machine Learning Model Benchmark Matrix

All models were evaluated using **Stratified 5-Fold Cross-Validation** with SMOTE imbalance treatment on training folds and tested on the untouched 20% holdout test set ($n=905$):

| Model Architecture | Hyperparameters | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Test PR-AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $C=1.0$, penalty='l2' | 84.64% | 39.81% | 80.77% | 0.5333 | 0.8842 | 0.5120 | Baseline |
| **Random Forest** | $n=200$, depth=12, split=4 | 88.51% | 50.00% | 54.81% | 0.5229 | 0.9142 | 0.5841 | Candidate |
| **XGBoost Classifier** | $n=180$, lr=0.08, depth=6 | **89.17%** | **48.48%** | **78.85%** | **0.6224** | **0.9261** | **0.6184** | **Champion** |

---

## Distributed Data Parallelism & Scalability Results

Testing 250,000 synthetic banking records across CPU execution partitions:

| Partitions | Execution Time (s) | Speedup Ratio | Throughput (Rows/Sec) | Parallel Efficiency |
| :---: | :---: | :---: | :---: | :---: |
| **1 Partition** | 0.0495s | 1.00x | 5,054,998 | 100.0% |
| **2 Partitions** | 0.0245s | **2.02x** | 10,211,709 | 100.0% |
| **4 Partitions** | 0.0202s | **2.45x** | 12,365,953 | 61.2% |
| **8 Partitions** | 0.0213s | 2.32x | 11,730,536 | 29.0% |

- **Broadcast Hash Join Optimization**: Broadcasted map-side joins eliminate cluster network shuffles on dimension lookups, accelerating joins by avoiding full distributed data redistribution.

---

## Business ROI & Financial Impact Analysis

Assuming **€10 per outbound call** and **€250 net customer lifetime value** per subscribed term deposit:
- **Blanket Mass Calling**: 4,521 calls $\times$ €10 = €45,210 cost; 521 conversions $\times$ €250 = €130,250 revenue $\implies$ **Net Profit = €85,040**.
- **Targeted Model Calling (Deciles 1–3)**: 1,500 calls $\times$ €10 = €15,000 cost; 420 conversions $\times$ €250 = €105,000 revenue $\implies$ **Net Profit = €90,000**.
- **Financial Result**: **+€4,960 direct profit lift with 67% fewer calls performed** (saving 3,021 calling attempts and eliminating customer brand annoyance).

---

## Quickstart & Execution Guide

### 1. Prerequisites & Environment Setup
```bash
# Clone the repository
git clone https://github.com/KARTHIKEYAN-31/Financial-Forecasting-Frontier-Distributed-ML.git
cd "Financial Forecasting Frontier Distributed ML"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Web Dashboard
```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501` to test the real-time propensity scorer and live transaction stream monitor.*

### 3. Run Distributed Big Data Scripts
```bash
# Run Spark Distributed EDA & MLlib Pipelines
python distributed_system/spark_eda_mllib.py

# Run Real-Time Spark Streaming & Anomaly Detection Simulation
python distributed_system/spark_streaming_realtime.py

# Run Data Parallelism & Broadcast Join Scaling Benchmarks
python distributed_system/data_parallelism_benchmark.py
```

### 4. Review Executed Notebooks
- [`Financial_Forecasting_EDA_Submission.ipynb`](Financial_Forecasting_EDA_Submission.ipynb)
- [`Financial_Forecasting_ML_Submission.ipynb`](Financial_Forecasting_ML_Submission.ipynb)
- [`Distributed_Banking_BigData_Spark_Hive.ipynb`](Distributed_Banking_BigData_Spark_Hive.ipynb)

---

## License & Author

**Author**: Karthikeyan (Data Science & Distributed ML Specialist)  
**Academic / Professional Capstone**: Financial Forecasting Frontier Distributed Machine Learning  
**Year**: 2026
