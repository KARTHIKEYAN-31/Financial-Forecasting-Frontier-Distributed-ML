"""
============================================================================
FINANCIAL FORECASTING FRONTIER: DISTRIBUTED ML & BANKING ANALYTICS
INTERACTIVE STREAMLIT DASHBOARD & REAL-TIME PREDICTION ENGINE
============================================================================
Author: Karthikeyan (Data Science & Distributed ML Specialist)
"""

import os
import sys
import time
import random
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page Configuration
st.set_page_config(
    page_title="Banking Financial Forecasting & Distributed ML",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 0.6rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1F2937;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Load Model Bundle
@st.cache_resource
def load_model_bundle():
    model_path = 'best_banking_model.joblib'
    if not os.path.exists(model_path):
        model_path = 'd:/project/woolf/Financial Forecasting Frontier Distributed ML/best_banking_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Load Dataset
@st.cache_data
def load_banking_data():
    dataset_path = 'dataset/dataset.csv'
    if not os.path.exists(dataset_path):
        dataset_path = 'd:/project/woolf/Financial Forecasting Frontier Distributed ML/dataset/dataset.csv'
    df = pd.read_csv(dataset_path)
    return df

bundle = load_model_bundle()
df = load_banking_data()

# Header Banner
st.markdown("<div class='main-header'>🏦 Financial Forecasting Frontier: Distributed Machine Learning</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Enterprise Retail Banking Analytics, Term Deposit Propensity Scoring & Real-Time Big Data Engine</div>", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Portfolio Dashboard",
    "🎯 Real-Time Propensity Scorer",
    "⚡ Real-Time Streaming & Anomaly Detection",
    "⚙️ Distributed Architecture & Big Data Benchmarks"
])

# ============================================================================
# TAB 1: EXECUTIVE PORTFOLIO DASHBOARD
# ============================================================================
with tab1:
    st.markdown("### 📈 Retail Banking Campaign Performance & Demographics")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    total_clients = len(df)
    total_subscribers = (df['y'] == 'yes').sum()
    conversion_rate = (total_subscribers / total_clients) * 100
    avg_balance = df['balance'].mean()
    
    with col1:
        st.markdown("<div class='card'><div class='metric-label'>Total Leads Contacted</div><div class='metric-value'>{:,}</div></div>".format(total_clients), unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'><div class='metric-label'>Term Deposit Subscribers</div><div class='metric-value'>{:,}</div></div>".format(total_subscribers), unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'><div class='metric-label'>Baseline Conversion Rate</div><div class='metric-value'>{:.2f}%</div></div>".format(conversion_rate), unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='card'><div class='metric-label'>Mean Account Balance</div><div class='metric-value'>€{:,.2f}</div></div>".format(avg_balance), unsafe_allow_html=True)
        
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### 🧑‍💼 Conversion Rate (%) by Occupational Role")
        job_stats = df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean() * 100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(x=job_stats.values, y=job_stats.index, palette='crest_r', ax=ax)
        ax.set_xlabel('Conversion Rate (%)')
        ax.set_ylabel('')
        ax.axvline(conversion_rate, color='red', linestyle='--', label=f'Avg ({conversion_rate:.1f}%)')
        ax.legend()
        st.pyplot(fig)
        
    with col_right:
        st.markdown("#### 📅 Monthly Campaign Seasonality (Volume vs Conversion)")
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        m_stats = df.groupby('month')['y'].agg(
            Volume='count',
            Conv=lambda x: (x == 'yes').mean() * 100
        ).reindex(month_order).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        color = '#4C72B0'
        ax1.bar(m_stats['month'], m_stats['Volume'], color=color, alpha=0.6, label='Call Volume')
        ax1.set_ylabel('Outbound Call Volume', color=color)
        ax2 = ax1.twinx()
        color = '#C44E52'
        ax2.plot(m_stats['month'], m_stats['Conv'], color=color, lw=2.5, marker='o', label='Conversion Rate (%)')
        ax2.set_ylabel('Conversion Rate (%)', color=color)
        ax2.grid(False)
        st.pyplot(fig)

    st.markdown("#### 💳 Joint Debt Burden Matrix (Housing Mortgages & Consumer Loans)")
    debt_df = df.groupby(['housing', 'loan'])['y'].apply(lambda x: (x == 'yes').mean() * 100).reset_index()
    debt_pivot = debt_df.pivot(index='housing', columns='loan', values='y')
    
    col_d1, col_d2 = st.columns([1, 2])
    with col_d1:
        st.dataframe(debt_pivot.style.format("{:.2f}%").background_gradient(cmap='YlGnBu'), use_container_width=True)
    with col_d2:
        st.info("💡 **Executive Takeaway**: Debt-free prospects convert at **16.88%**, whereas clients with dual debt (housing + personal loan) convert at only **6.16%**. Filtering out multi-loan borrowers eliminates low-yield calling overhead.")

# ============================================================================
# TAB 2: REAL-TIME PROPENSITY SCORER
# ============================================================================
with tab2:
    st.markdown("### 🎯 Real-Time Term Deposit Propensity Scorer")
    st.markdown("Enter prospective client demographic and communication parameters to compute the real-time deposit conversion probability using the trained **XGBoost Production Engine**.")
    
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        inp_age = st.slider("Customer Age", min_value=18, max_value=90, value=35)
        inp_job = st.selectbox("Profession", sorted(df['job'].unique()), index=sorted(df['job'].unique()).index('management'))
        inp_marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
        inp_education = st.selectbox("Education Level", ['primary', 'secondary', 'tertiary', 'unknown'], index=2)
        inp_default = st.selectbox("Credit in Default?", ['no', 'yes'])
        
    with col_input2:
        inp_balance = st.number_input("Account Balance (€)", min_value=-5000, max_value=100000, value=2500)
        inp_housing = st.selectbox("Housing Mortgage Loan?", ['no', 'yes'], index=0)
        inp_loan = st.selectbox("Personal Consumer Loan?", ['no', 'yes'], index=0)
        inp_contact = st.selectbox("Contact Communication Channel", ['cellular', 'telephone', 'unknown'], index=0)
        inp_month = st.selectbox("Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], index=9)
        
    with col_input3:
        inp_day = st.slider("Day of Month", min_value=1, max_value=31, value=15)
        inp_duration = st.slider("Call Duration (Seconds)", min_value=10, max_value=1500, value=380)
        inp_campaign = st.slider("Campaign Contacts Performed", min_value=1, max_value=15, value=1)
        inp_pdays = st.number_input("Days Since Prior Contact (-1 = Never)", min_value=-1, max_value=999, value=180)
        inp_previous = st.number_input("Previous Contacts Count", min_value=0, max_value=30, value=2)
        inp_poutcome = st.selectbox("Previous Campaign Outcome", ['unknown', 'failure', 'other', 'success'], index=3)

    if st.button("🚀 Score Customer Propensity", type="primary"):
        if bundle is not None:
            model = bundle['model']
            scaler = bundle['scaler']
            feature_names = bundle['feature_names']
            num_scale_cols = bundle['num_scale_cols']
            
            # Construct single-row DataFrame
            row_dict = {
                'age': inp_age,
                'job': inp_job,
                'marital': inp_marital,
                'education': inp_education,
                'default': inp_default,
                'balance': inp_balance,
                'housing': inp_housing,
                'loan': inp_loan,
                'contact': inp_contact,
                'day': inp_day,
                'month': inp_month,
                'duration': inp_duration,
                'campaign': inp_campaign,
                'pdays': inp_pdays,
                'previous': inp_previous,
                'poutcome': inp_poutcome
            }
            input_df = pd.DataFrame([row_dict])
            
            # Apply preprocessing
            binary_map = {'no': 0, 'yes': 1}
            input_df['default'] = input_df['default'].map(binary_map)
            input_df['housing'] = input_df['housing'].map(binary_map)
            input_df['loan'] = input_df['loan'].map(binary_map)
            input_df['education'] = input_df['education'].map({'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3})
            
            input_encoded = pd.get_dummies(input_df, columns=['job', 'marital', 'contact', 'month', 'poutcome'], drop_first=False, dtype=int)
            
            # Add engineered features
            input_encoded['balance_to_age'] = input_encoded['balance'] / input_encoded['age']
            input_encoded['was_contacted_prev'] = (input_encoded['pdays'] != -1).astype(int)
            input_encoded['total_debt_count'] = input_encoded['housing'] + input_encoded['loan']
            input_encoded['balance_log'] = np.sign(input_encoded['balance']) * np.log1p(np.abs(input_encoded['balance']))
            input_encoded['duration_log'] = np.log1p(input_encoded['duration'])
            
            # Align with model feature columns
            for col in feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[feature_names]
            
            # Scale numerical features
            input_encoded[num_scale_cols] = scaler.transform(input_encoded[num_scale_cols])
            
            # Predict
            prob = model.predict_proba(input_encoded)[0, 1]
            pred_class = 1 if prob >= 0.5 else 0
            
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown("#### 🎯 Prediction Results")
                if prob >= 0.70:
                    st.success(f"### Tier 1: High Propensity ({prob*100:.1f}%)")
                    st.markdown("✅ **Action**: Prioritize for immediate VIP Senior Relationship Manager outreach.")
                elif prob >= 0.35:
                    st.warning(f"### Tier 2: Moderate Propensity ({prob*100:.1f}%)")
                    st.markdown("⚡ **Action**: Queue for standard telemarketing with tailored product script.")
                else:
                    st.error(f"### Tier 3: Low Propensity ({prob*100:.1f}%)")
                    st.markdown("⛔ **Action**: Suppress outbound calls. Channel to digital self-serve campaigns.")
                    
            with res_col2:
                st.markdown("#### 📊 Key Contributing Decision Factors")
                factors = []
                if inp_duration > 300:
                    factors.append(("Call Duration (>5 mins)", "+ Strong Positive", "green"))
                if inp_poutcome == 'success':
                    factors.append(("Previous Campaign Success", "+ Extreme Positive", "green"))
                if inp_housing == 'no' and inp_loan == 'no':
                    factors.append(("Debt-Free Financial Standing", "+ Moderate Positive", "green"))
                if inp_contact == 'cellular':
                    factors.append(("Cellular Communication Channel", "+ Positive", "green"))
                if inp_housing == 'yes':
                    factors.append(("Active Housing Mortgage", "- Negative Drag", "red"))
                if inp_campaign > 3:
                    factors.append(("High Contact Frequency (>3)", "- Diminishing Return", "red"))
                    
                for factor, impact, color in factors:
                    if color == 'green':
                        st.markdown(f"- **{factor}**: `{impact}`")
                    else:
                        st.markdown(f"- **{factor}**: `{impact}`")
        else:
            st.error("Model bundle not found. Please train models first.")

# ============================================================================
# TAB 3: REAL-TIME STREAMING & ANOMALY DETECTION
# ============================================================================
with tab3:
    st.markdown("### ⚡ Real-Time Spark Structured Streaming & Anomaly Monitoring")
    st.markdown("Simulate high-frequency transaction event ingestion with real-time sliding/tumbling window aggregations and automated fraud alert triggering.")
    
    stream_col1, stream_col2 = st.columns([1, 3])
    with stream_col1:
        event_count = st.slider("Simulated Events to Ingest", min_value=10, max_value=100, value=30, step=10)
        start_stream = st.button("▶️ Launch Live Event Stream", type="primary")
        
    if start_stream:
        from distributed_system.spark_streaming_realtime import RealTimeTransactionGenerator
        gen = RealTimeTransactionGenerator()
        
        stream_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        events = []
        alerts = []
        
        for i in range(1, event_count + 1):
            ev = gen.generate_event()
            events.append(ev)
            
            if ev['amount_eur'] > 2000.0 or ev['device_risk'] in ['High', 'Critical']:
                alerts.append({
                    'Alert_ID': f"ALT-{len(alerts)+1:03d}",
                    'Txn_ID': ev['transaction_id'],
                    'Customer': ev['customer_id'],
                    'Amount_EUR': f"€{ev['amount_eur']:,.2f}",
                    'Merchant': ev['merchant'],
                    'Location': ev['location'],
                    'Severity': 'CRITICAL FRAUD' if ev['amount_eur'] > 5000 else 'HIGH RISK'
                })
                
            stream_df = pd.DataFrame(events).tail(10)
            with stream_placeholder.container():
                st.markdown(f"**Ingested Stream Buffer ({i}/{event_count} transactions)**")
                st.dataframe(stream_df[['transaction_id', 'timestamp', 'customer_id', 'amount_eur', 'merchant', 'location', 'device_risk']], use_container_width=True)
                
            time.sleep(0.05)
            
        st.success(f"Stream simulation completed: Ingested {len(events)} events | Total Volume: €{sum([e['amount_eur'] for e in events]):,.2f}")
        
        if alerts:
            st.error(f"🚨 {len(alerts)} High-Risk Anomaly Alerts Intercepted!")
            st.table(pd.DataFrame(alerts))
        else:
            st.info("No critical anomaly spikes detected in this micro-batch.")

# ============================================================================
# TAB 4: DISTRIBUTED ARCHITECTURE & BIG DATA BENCHMARKS
# ============================================================================
with tab4:
    st.markdown("### ⚙️ Distributed Architecture, HiveQL Warehousing & Parallelism Benchmarks")
    
    arch_col1, arch_col2 = st.columns(2)
    
    with arch_col1:
        st.markdown("#### 🏗️ Big Data Pipeline Architecture")
        st.code("""
[Raw Telemarketing Logs] 
         │
         ▼
[Hadoop HDFS Storage Cluster] 
         │
         ▼
[Apache Hive Warehouse (ORC / Partitioned / Bucketed)]
         │
    ┌────┴────────────────────────┐
    ▼                             ▼
[Spark SQL & Distributed EDA]  [Spark MLlib Distributed Pipelines]
    │                             │
    ▼                             ▼
[Sliding Window Streaming Engine]  [Tuned XGBoost / RF Classifiers]
        """, language="text")
        
    with arch_col2:
        st.markdown("#### ⚡ Data Parallelism Benchmark (Partition Scaling)")
        bench_data = pd.DataFrame({
            'Partitions': [1, 2, 4, 8],
            'Execution_Time_s': [0.0495, 0.0245, 0.0202, 0.0213],
            'Speedup': [1.00, 2.02, 2.45, 2.32],
            'Throughput_Rows_Sec': [5054998, 10211709, 12365953, 11730536]
        })
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(bench_data['Partitions'], bench_data['Speedup'], marker='o', lw=2.5, color='crimson', label='Observed Speedup')
        ax.plot(bench_data['Partitions'], bench_data['Partitions'], linestyle='--', color='grey', label='Linear Speedup')
        ax.set_xlabel('Partitions / Worker Threads')
        ax.set_ylabel('Speedup Factor (x)')
        ax.set_title('Speedup Scaling across Distributed Partitions')
        ax.legend()
        st.pyplot(fig)
        
    st.markdown("#### 📋 Hive Warehouse DDL Schema Specification")
    st.code("""
CREATE TABLE IF NOT EXISTS bank_marketing_partitioned_bucketed (
    age INT, marital STRING, education STRING, default STRING,
    balance INT, housing STRING, loan STRING, contact STRING,
    day INT, duration INT, campaign INT, pdays INT, previous INT,
    poutcome STRING, y STRING, total_debt_count INT
)
PARTITIONED BY (month STRING, job STRING)
CLUSTERED BY (age) SORTED BY (balance DESC) INTO 6 BUCKETS
STORED AS ORC
TBLPROPERTIES ("orc.compress"="SNAPPY");
    """, language="sql")

st.sidebar.markdown("### 📌 Capstone Deliverables")
st.sidebar.markdown("- 📓 `Financial_Forecasting_EDA_Submission.ipynb`")
st.sidebar.markdown("- 📓 `Financial_Forecasting_ML_Submission.ipynb`")
st.sidebar.markdown("- 📓 `Distributed_Banking_BigData_Spark_Hive.ipynb`")
st.sidebar.markdown("- 📦 `best_banking_model.joblib`")
st.sidebar.markdown("- 📄 `docs/01_FINAL_PROJECT_REPORT.md`")
st.sidebar.markdown("- 📄 `README.md`")
st.sidebar.markdown("---")
st.sidebar.caption("Financial Forecasting Frontier | Distributed Machine Learning © 2026")
