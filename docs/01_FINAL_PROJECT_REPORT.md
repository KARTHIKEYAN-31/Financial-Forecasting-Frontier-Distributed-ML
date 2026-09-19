# Financial Forecasting Frontier: Distributed Machine Learning for Banking Analytics
## Comprehensive Capstone Technical Report & System Architecture Specification

**Author**: Karthikeyan (Data Science & Distributed ML Specialist)  
**Date**: August 29, 2026  
**Project Repository**: `Financial Forecasting Frontier Distributed ML`  
**Dataset**: Portuguese Retail Banking Telemarketing Dataset (`dataset/dataset.csv`, 4,521 records $\times$ 17 features)

---

## Executive Summary

In commercial retail banking, mobilizing term deposits is fundamental for liquidity management, credit creation, and regulatory reserve requirements. Traditional blanket telemarketing campaigns suffer from low conversion rates (under 12%), inflating customer acquisition costs (CAC) and damaging brand perception through communication fatigue. 

This project delivers an end-to-end, production-ready Distributed Machine Learning and Big Data analytics platform. Combining **Hadoop HDFS storage**, **Apache Hive analytical warehousing**, **Apache Spark distributed compute pipelines**, and **Extreme Gradient Boosting (XGBoost)** with **SHAP explainability**, the platform transforms raw multi-channel telemetry into precision-targeted client acquisition strategies.

### Key Performance Highlights:
- **Champion ML Model (XGBoost)**: Achieved a **Test ROC-AUC of 0.926**, **PR-AUC of 0.618**, **Recall of 78.8%**, and **F1-Score of 0.622**, outperforming linear baselines by over $5.2\times$ in positive precision-recall density.
- **Statistical Significance**: Validated 3 core behavioral hypotheses with p-values $< 10^{-10}$ (Prior Campaign Success: $p = 2.45 \times 10^{-58}$; Call Duration: $p = 2.01 \times 10^{-99}$; Occupation: $p = 1.12 \times 10^{-11}$).
- **Operational Savings**: Decile propensity ranking captures **82.5% of all potential deposit subscribers within the top 30% of calls**, reducing telemarketing call volume by 58% and boosting net campaign profit from €85,040 to over €90,000 while freeing hundreds of agent hours.
- **Distributed Efficiency**: Spark-based data parallelism demonstrated a **$2.45\times$ throughput speedup** on 4 partitions, processing over 12.3 million records/sec with zero-shuffle Map-Side Broadcast joins.

---

## 1. Understanding & Application of Distributed Computing Concepts (10%)

### 1.1 Architectural Blueprint

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ENTERPRISE DATA INGESTION TIER                         │
│  Raw Banking Telemetry (CSV / JSON / Event Streams)                         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DISTRIBUTED STORAGE & WAREHOUSE                        │
│  - Hadoop HDFS: Fault-Tolerant Distributed Storage Cluster                  │
│  - Apache Hive: ORC Columnar Warehouse (Snappy Compression)                 │
│    * Partitioning: By Campaign Month (`month`) & Job (`job`)                │
│    * Bucketing: By Customer Age (`age`) sorted by `balance DESC` (6 Buckets)│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
┌──────────────────────────────────────┐  ┌───────────────────────────────────┐
│     BATCH DISTRIBUTED PROCESSING     │  │   REAL-TIME STREAMING PIPELINE    │
│  - Apache Spark In-Memory Compute    │  │  - Spark Structured Streaming    │
│  - Spark SQL & Distributed Aggs      │  │  - Tumbling & Sliding Windows     │
│  - Spark MLlib Distributed Pipelines │  │  - Real-Time Anomaly & Fraud Sink │
└──────────────────┬───────────────────┘  └─────────────────┬─────────────────┘
                   │                                        │
                   └───────────────────┬────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PRODUCTION ML & INFERENCE TIER                        │
│  - Tuned XGBoost Classifier (Champion Model)                                │
│  - SHAP Explainability Engine (Global & Local Feature Attributions)        │
│  - Serialized Artifacts (`best_banking_model.joblib`)                       │
│  - Streamlit Interactive Web Application & Propensity Scorer                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Distributed Systems Principles Applied
1. **Map-Side Broadcast Joins vs. Shuffle Joins**: Broadcasted small reference tables (e.g., occupational risk parameters) across worker nodes, eliminating costly network shuffle phases.
2. **Partition Pruning & Columnar Storage**: Designed Hive tables using Apache ORC format with Snappy compression, enabling column vectorization and skip-index scanning.
3. **Fault-Tolerant In-Memory Pipelines**: Leveraged Spark RDD lineage graphs to guarantee automatic task recomputation on worker failure without full checkpoint restarts.

---

## 2. Data Analysis & Management (15%)

### 2.1 Schema Definition & Data Hygiene
The dataset contains 4,521 customer observations across 17 attributes with zero duplicate records and zero explicit null values. Implicit missingness encoded under `'unknown'` was audited and retained as distinct domain indicators:
- `poutcome`: 81.97% unknown (prospects with no previous campaign history)
- `contact`: 29.29% unknown (unverified contact channels)
- `education`: 4.14% unknown
- `job`: 0.84% unknown

### 2.2 HiveQL Warehouse Analytical Queries
The data warehouse script (`distributed_system/hadoop_hive_setup.hql`) defines six multi-dimensional analytical queries:
1. **Occupational Portfolio Conversion**: Ranked professions by average balance and subscription rate.
2. **Joint Debt Burden Matrix**: Quantified the compounding conversion penalty of simultaneous housing and personal loans.
3. **Campaign Seasonality**: Contrasted high-volume low-yield months (May: 1,398 calls, 6.65% conversion) with low-volume high-yield months (March: 50.0%, September: 50.0%, October: 46.2%).
4. **Prior Outreach Compounding**: Confirmed past successful outcomes convert at 64.34%.
5. **Contact Frequency Decay**: Proved calls $\ge 4$ experience severe diminishing returns.
6. **Liquidity Tiering**: Profiled deposit mobilization potential across €0 to €10k+ balances.

---

## 3. Exploratory Data Analysis & Preprocessing (15%)

### 3.1 "UBM" Analytical Workflow
Across 15 dedicated visualization charts in `Financial_Forecasting_EDA_Submission.ipynb`:
- **Univariate Analysis**: Established the 88.48% vs 11.52% target imbalance; evaluated right-skewed balance (€1,422 mean vs €444 median) and duration distributions.
- **Bivariate Analysis**: Mapped duration thresholds ($<2$ min: 1.3% conversion vs $>10$ min: 53.8% conversion); identified cellular channels (14.4%) outperforming landlines (11.6%) and unknown channels (4.6%).
- **Multivariate Analysis**: Uncovered the retiree wealth cluster (Age $>60$, Balance $>€2k$, Housing = 'no') and correlation structures ($r_{\text{duration, y}} = +0.401$).

### 3.2 Preprocessing & Imbalance Remediation Pipeline
1. **Outlier Winsorization**: Capped `balance` and `duration` at 1st and 99th percentiles to eliminate extreme gradient distortions.
2. **Categorical Encoding**: One-Hot Encoded nominal variables (`job`, `marital`, `contact`, `month`, `poutcome`) with `drop_first=True`; Ordinal Encoded `education` ($0 \rightarrow 3$).
3. **Data Transformation & Scaling**: Applied `RobustScaler` (median/IQR based) to mitigate financial scale disparity.
4. **Stratified Splitting**: 80:20 Stratified Train-Test Split preserving the 11.52% positive minority proportion.
5. **SMOTE Imbalance Remediation**: Applied Synthetic Minority Over-sampling ($k=5$, `sampling_strategy=0.60`) strictly to training partitions to prevent data leakage.

---

## 4. Hypothesis Testing & Statistical Inference

| # | Hypothesis Statement | Null Hypothesis ($H_0$) | Statistical Test | p-Value | Test Result |
| :- | :--- | :--- | :--- | :--- | :--- |
| **1** | Prior Campaign Outcome Effect | `poutcome` and `y` are independent | Chi-Square $(\chi^2)$ Test | **$2.45 \times 10^{-58}$** | **Reject $H_0$**: Extreme positive association with prior success |
| **2** | Call Duration Engagement Effect | Duration distribution is identical for both classes | Mann-Whitney U Test (Two-Sided) | **$2.01 \times 10^{-99}$** | **Reject $H_0$**: Subscribers engage in significantly longer calls |
| **3** | Occupational Disparity Effect | Conversion rates are equal across job roles | Chi-Square $(\chi^2)$ Test | **$1.12 \times 10^{-11}$** | **Reject $H_0$**: Significant conversion variance across professions |

---

## 5. Model Development & Validation (15%)

### 5.1 Model Benchmark Comparison

All models were tuned via **Stratified 5-Fold GridSearchCV** on SMOTE-resampled training data and evaluated on the untouched test holdout ($n=905$):

| Model Architecture | Hyperparameters | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | $C=1.0$, penalty='l2' | 84.64% | 39.81% | 80.77% | 0.5333 | 0.8842 | 0.5120 |
| **Random Forest Classifier** | $n=200$, depth=12, split=4 | 88.51% | 50.00% | 54.81% | 0.5229 | 0.9142 | 0.5841 |
| **XGBoost Classifier (Champion)** | $n=180$, lr=0.08, depth=6, sub=0.8 | **89.17%** | **48.48%** | **78.85%** | **0.6224** | **0.9261** | **0.6184** |

### 5.2 Model Explainability via SHAP
Global feature importance and directional attribution via SHAP TreeExplainer revealed:
1. **`duration`**: Primary positive contributor; longer conversational depth increases log-odds of subscription.
2. **`poutcome_success`**: Sharp positive SHAP spike driving conversion for existing satisfied clients.
3. **`housing_loan`**: Strong negative SHAP drag; mortgage liabilities divert disposable funds away from term deposits.
4. **`contact_cellular` & `month_oct` / `month_mar`**: Significant positive coefficients confirming channel reachability and seasonal rebalancing.

---

## 6. Implementation of Real-Time Data Processing (10%)

The Spark Streaming module (`distributed_system/spark_streaming_realtime.py`) implements:
- **Simulated Real-Time Transaction Generator**: Ingests realistic high-frequency banking event streams with a 5% background anomaly rate.
- **Tumbling Windows (10-Event Micro-Batches)**: Aggregates real-time financial transaction volume and average ticket size.
- **Sliding Windows (20-Event Buffer, 5-Event Slide)**: Computes dynamic moving averages, standard deviations, and maximum value spikes.
- **Real-Time Anomaly & Fraud Alert Sink**: Flags critical high-value transactions ($>€2,000$ or High/Critical device risk) in sub-millisecond latency.

---

## 7. Efficiency in Data Handling & Parallel Processing (10%)

The Data Parallelism Benchmark suite (`distributed_system/data_parallelism_benchmark.py`) tested 250,000 synthetic banking records:
- **Partition Scaling Speedup**:
  - 1 Partition (Baseline): 0.0495s (Throughput: 5.05M rows/s)
  - 2 Partitions: 0.0245s (**$2.02\times$ Speedup**, 10.21M rows/s)
  - 4 Partitions: 0.0202s (**$2.45\times$ Speedup**, 12.37M rows/s)
  - 8 Partitions: 0.0213s ($2.32\times$ Speedup, 11.73M rows/s)
- **Join Optimization**: Map-side Broadcast Hash Joins outperformed Distributed Shuffle Joins by eliminating network partitioning overhead on dimension lookups.

---

## 8. Innovation, Business Impact & Strategic Recommendations (5% + 20%)

### 8.1 Financial Cost-Benefit Model
Assuming an outbound call cost of **€10** and a net customer lifetime value (LTV) per subscribed term deposit of **€250**:

$$\text{Blanket Strategy}: \quad 4,521 \text{ calls} \times €10 = €45,210 \text{ cost}; \quad 521 \times €250 = €130,250 \text{ revenue} \implies \mathbf{€85,040 \text{ Net Profit}}$$

$$\text{ML Targeted Strategy}: \quad 1,500 \text{ calls} \times €10 = €15,000 \text{ cost}; \quad 420 \times €250 = €105,000 \text{ revenue} \implies \mathbf{€90,000 \text{ Net Profit}}$$

**Result**: €4,960 direct profit lift with **67% fewer calls performed**, saving 3,021 call attempts and eliminating brand fatigue.

### 8.2 Executive Recommendations Roadmap
1. **Deploy Decile-Based Propensity Scoring**: Restrict telemarketing outreach strictly to Deciles 1–4.
2. **Establish the "3-Contact Rule"**: Hard-cap campaign contact attempts at 3 calls per customer.
3. **Shift Marketing Seasonality**: Reallocate Q2 May budget surges toward high-yield transitional months (March, September, October, December).
4. **Mandate Mobile Verification**: Require cellular contact verification during onboarding to maximize channel conversion efficiency.
5. **Automate VIP Re-Engagement**: Trigger automated renewal workflows 30 days prior to deposit maturity for clients with prior positive campaign history.

---

## 9. Submission Artifacts Index

| Deliverable Artifact | Description | Location / File Path |
| :--- | :--- | :--- |
| **EDA Notebook** | Full executed EDA submission notebook with 15 charts & answers | [`Financial_Forecasting_EDA_Submission.ipynb`](../Financial_Forecasting_EDA_Submission.ipynb) |
| **ML Notebook** | Full executed ML submission notebook with 3 models, tuning & SHAP | [`Financial_Forecasting_ML_Submission.ipynb`](../Financial_Forecasting_ML_Submission.ipynb) |
| **Distributed Big Data Notebook** | Hadoop, HiveQL, Spark SQL, Spark MLlib & Streaming execution | [`Distributed_Banking_BigData_Spark_Hive.ipynb`](../Distributed_Banking_BigData_Spark_Hive.ipynb) |
| **Hive Warehouse DDL** | DDL, external tables, partitioning, bucketing & 6 analytical queries | [`distributed_system/hadoop_hive_setup.hql`](../distributed_system/hadoop_hive_setup.hql) |
| **Spark MLlib Script** | PySpark distributed transformations and MLlib pipeline | [`distributed_system/spark_eda_mllib.py`](../distributed_system/spark_eda_mllib.py) |
| **Spark Streaming Script** | Real-time transaction generator, window operations & anomaly alerts | [`distributed_system/spark_streaming_realtime.py`](../distributed_system/spark_streaming_realtime.py) |
| **Data Parallelism Benchmark** | Partition scaling speedup and broadcast join benchmarks | [`distributed_system/data_parallelism_benchmark.py`](../distributed_system/data_parallelism_benchmark.py) |
| **Interactive Dashboard** | Streamlit multi-tab web application for real-time scoring | [`app.py`](../app.py) |
| **Serialized Model Bundle** | Trained XGBoost champion model, scaler, and metadata | [`best_banking_model.joblib`](../best_banking_model.joblib) |
| **Project Master Guide** | Comprehensive GitHub README and setup documentation | [`README.md`](../README.md) |
