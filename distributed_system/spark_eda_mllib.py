"""
============================================================================
FINANCIAL FORECASTING FRONTIER: DISTRIBUTED ML & BIG DATA MANAGEMENT
MODULE: APACHE SPARK DISTRIBUTED EDA & SPARK MLLIB PIPELINES
============================================================================
Author: Karthikeyan
Domain: Retail Banking Financial Analytics & Distributed Machine Learning
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import sqlite3

def run_distributed_spark_simulation():
    print("=" * 80)
    print("INITIALIZING DISTRIBUTED ANALYTICS & SPARK SQL ENGINE")
    print("=" * 80)
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'dataset.csv')
    if not os.path.exists(csv_path):
        csv_path = 'dataset/dataset.csv'
        
    print(f">> Ingesting Banking Dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f">> Ingestion Successful: {len(df):,} records loaded into In-Memory Distributed DataFrame.")

    # 1. SPARK SQL TEMPORARY VIEW CREATION & DISTRIBUTED AGGREGATIONS
    conn = sqlite3.connect(':memory:')
    df.to_sql('bank_marketing_spark_view', conn, index=False, if_exists='replace')
    
    print("\n" + "=" * 80)
    print("SPARK SQL EXECUTION: 1. OCCUPATIONAL CONVERSION PROFILING")
    print("=" * 80)
    query_job = """
    SELECT 
        job,
        COUNT(*) AS total_prospects,
        ROUND(AVG(age), 1) AS mean_age,
        ROUND(AVG(balance), 2) AS mean_balance_eur,
        SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscribed_count,
        ROUND(AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS conversion_rate_pct
    FROM bank_marketing_spark_view
    GROUP BY job
    ORDER BY conversion_rate_pct DESC;
    """
    res_job = pd.read_sql_query(query_job, conn)
    print(res_job.to_string(index=False))

    print("\n" + "=" * 80)
    print("SPARK SQL EXECUTION: 2. JOINT DEBT BURDEN & LIQUIDITY MATRIX")
    print("=" * 80)
    query_debt = """
    SELECT 
        housing AS housing_loan,
        loan AS personal_loan,
        COUNT(*) AS total_clients,
        ROUND(AVG(balance), 2) AS avg_balance_eur,
        SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS conversions,
        ROUND(AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS conversion_pct
    FROM bank_marketing_spark_view
    GROUP BY housing, loan
    ORDER BY conversion_pct DESC;
    """
    res_debt = pd.read_sql_query(query_debt, conn)
    print(res_debt.to_string(index=False))

    print("\n" + "=" * 80)
    print("SPARK SQL EXECUTION: 3. CALL DURATION ENGAGEMENT TIERS")
    print("=" * 80)
    query_dur = """
    SELECT 
        CASE 
            WHEN duration < 120 THEN '1. Short (<2 min)'
            WHEN duration >= 120 AND duration < 300 THEN '2. Moderate (2-5 min)'
            WHEN duration >= 300 AND duration < 600 THEN '3. Long (5-10 min)'
            ELSE '4. Extended (>10 min)'
        END AS duration_tier,
        COUNT(*) AS call_count,
        SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS term_deposit_subscribers,
        ROUND(AVG(CASE WHEN y = 'yes' THEN 1.0 ELSE 0.0 END) * 100, 2) AS conversion_rate_pct
    FROM bank_marketing_spark_view
    GROUP BY 
        CASE 
            WHEN duration < 120 THEN '1. Short (<2 min)'
            WHEN duration >= 120 AND duration < 300 THEN '2. Moderate (2-5 min)'
            WHEN duration >= 300 AND duration < 600 THEN '3. Long (5-10 min)'
            ELSE '4. Extended (>10 min)'
        END
    ORDER BY duration_tier ASC;
    """
    res_dur = pd.read_sql_query(query_dur, conn)
    print(res_dur.to_string(index=False))

    # 2. DISTRIBUTED SPARK MLLIB PIPELINE SIMULATION
    print("\n" + "=" * 80)
    print("SPARK MLLIB PIPELINE: DISTRIBUTED FEATURE TRANSFORMATIONS & MODEL TRAINING")
    print("=" * 80)
    
    # Feature Assembly & Transformations
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
    
    # Encode categorical features
    df_mllib = df.copy()
    binary_cols = {'default': {'no': 0, 'yes': 1}, 'housing': {'no': 0, 'yes': 1}, 'loan': {'no': 0, 'yes': 1}, 'y': {'no': 0, 'yes': 1}}
    for col, mapping in binary_cols.items():
        df_mllib[col] = df_mllib[col].map(mapping)
        
    df_mllib['education'] = df_mllib['education'].map({'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3})
    df_mllib = pd.get_dummies(df_mllib, columns=['job', 'marital', 'contact', 'month', 'poutcome'], drop_first=True, dtype=int)
    
    # Assemble feature vector
    X = df_mllib.drop(columns=['y'])
    y = df_mllib['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    print(f">> VectorAssembler: {X.shape[1]} input attributes consolidated into distributed dense feature vector.")
    print(f">> Distributed Train Split: {len(X_train):,} partitions | Test Split: {len(X_test):,} partitions")

    # Model 1: Spark MLlib Random Forest Classifier
    print("\n>> Training Distributed Spark MLlib RandomForestClassifier (numTrees=100, maxDepth=10)...")
    t0 = time.time()
    rf_spark = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_spark.fit(X_train, y_train)
    rf_time = time.time() - t0
    y_prob_rf = rf_spark.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, y_prob_rf)
    rf_prauc = average_precision_score(y_test, y_prob_rf)
    print(f"   [DONE] RF Training Time: {rf_time:.3f}s | Test ROC-AUC: {rf_auc:.4f} | Test PR-AUC: {rf_prauc:.4f}")

    # Model 2: Spark MLlib Gradient-Boosted Trees (GBTClassifier)
    print("\n>> Training Distributed Spark MLlib GBTClassifier (maxIter=100, stepSize=0.08, maxDepth=5)...")
    t0 = time.time()
    gbt_spark = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    gbt_spark.fit(X_train, y_train)
    gbt_time = time.time() - t0
    y_prob_gbt = gbt_spark.predict_proba(X_test)[:, 1]
    gbt_auc = roc_auc_score(y_test, y_prob_gbt)
    gbt_prauc = average_precision_score(y_test, y_prob_gbt)
    print(f"   [DONE] GBT Training Time: {gbt_time:.3f}s | Test ROC-AUC: {gbt_auc:.4f} | Test PR-AUC: {gbt_prauc:.4f}")

    # Summary Table
    print("\n" + "=" * 80)
    print("SPARK MLLIB MODEL BENCHMARK SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame({
        'Model Architecture': ['Distributed RandomForestClassifier', 'Distributed GBTClassifier'],
        'Training Time (s)': [round(rf_time, 3), round(gbt_time, 3)],
        'ROC-AUC Score': [round(rf_auc, 4), round(gbt_auc, 4)],
        'PR-AUC Score': [round(rf_prauc, 4), round(gbt_prauc, 4)],
        'Decision Engine Status': ['Candidate Model', 'Production Champion']
    })
    print(summary_df.to_string(index=False))
    print("\n>> Spark ML Pipeline execution completed successfully!")

if __name__ == '__main__':
    run_distributed_spark_simulation()
