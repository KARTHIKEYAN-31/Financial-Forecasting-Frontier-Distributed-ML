-- ============================================================================
-- FINANCIAL FORECASTING FRONTIER: DISTRIBUTED ML & BIG DATA MANAGEMENT
-- HADOOP HDFS INGESTION & APACHE HIVE ANALYTICAL WAREHOUSE SETUP
-- ============================================================================

-- 1. CREATE DEDICATED WAREHOUSE DATABASE
CREATE DATABASE IF NOT EXISTS retail_banking_dw
COMMENT 'Enterprise Data Warehouse for Retail Banking & Campaign Analytics'
WITH DBPROPERTIES ('creator' = 'Karthikeyan', 'created_on' = '2026-08-29');

USE retail_banking_dw;

-- 2. CREATE HIVE EXTERNAL TABLE FOR RAW DATA INGESTION FROM HDFS
-- Simulates raw CSV ingestion from distributed Hadoop HDFS storage path
DROP TABLE IF EXISTS raw_bank_marketing_ext;
CREATE EXTERNAL TABLE IF NOT EXISTS raw_bank_marketing_ext (
    age INT COMMENT 'Customer age in years',
    job STRING COMMENT 'Profession or occupational classification',
    marital STRING COMMENT 'Marital status (married, single, divorced)',
    education STRING COMMENT 'Educational attainment level',
    default STRING COMMENT 'Credit default status (no, yes)',
    balance INT COMMENT 'Average yearly account balance in Euros',
    housing STRING COMMENT 'Housing mortgage status (no, yes)',
    loan STRING COMMENT 'Personal consumer loan status (no, yes)',
    contact STRING COMMENT 'Communication contact type (cellular, telephone, unknown)',
    day INT COMMENT 'Last contact day of the month',
    month STRING COMMENT 'Last contact month of the year',
    duration INT COMMENT 'Last contact duration in seconds',
    campaign INT COMMENT 'Number of contacts during current campaign',
    pdays INT COMMENT 'Days since client was contacted in previous campaign',
    previous INT COMMENT 'Number of contacts performed before this campaign',
    poutcome STRING COMMENT 'Outcome of previous marketing campaign',
    y STRING COMMENT 'Term deposit subscription target (no, yes)'
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/hadoop/banking/raw_data/'
TBLPROPERTIES ("skip.header.line.count"="1");

-- 3. OPTIMIZED MANAGED TABLE WITH ORC FORMAT, PARTITIONING & BUCKETING
-- Partitioned by 'month' for partition pruning during seasonal analytics
-- Bucketed by 'job' into 6 buckets with sorted 'age' for efficient Map-Side bucketed joins
DROP TABLE IF EXISTS bank_marketing_partitioned_bucketed;
CREATE TABLE IF NOT EXISTS bank_marketing_partitioned_bucketed (
    age INT,
    marital STRING,
    education STRING,
    default STRING,
    balance INT,
    housing STRING,
    loan STRING,
    contact STRING,
    day INT,
    duration INT,
    campaign INT,
    pdays INT,
    previous INT,
    poutcome STRING,
    y STRING,
    total_debt_count INT
)
PARTITIONED BY (month STRING, job STRING)
CLUSTERED BY (age) SORTED BY (balance DESC) INTO 6 BUCKETS
STORED AS ORC
TBLPROPERTIES (
    "orc.compress"="SNAPPY",
    "transactional"="false",
    "comment"="Optimized columnar storage for high-velocity SQL analytical querying"
);

-- Enable dynamic partitioning configuration
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.enforce.bucketing=true;
SET hive.exec.max.dynamic.partitions=1000;
SET hive.exec.max.dynamic.partitions.pernode=500;

-- 4. ENTERPRISE HIVESQL ANALYTICAL QUERIES

-- QUERY 1: Portfolio Executive Summary & Conversion by Profession
-- Computes key financial KPIs, average liquid balance, and term deposit adoption rate
SELECT 
    job,
    COUNT(*) AS total_clients,
    ROUND(AVG(age), 1) AS avg_age,
    ROUND(AVG(balance), 2) AS avg_balance_eur,
    ROUND(PERCENTILE_APPROX(balance, 0.5), 2) AS median_balance_eur,
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscribed_count,
    ROUND((SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS conversion_rate_pct
FROM raw_bank_marketing_ext
GROUP BY job
ORDER BY conversion_rate_pct DESC;

-- QUERY 2: Debt Liability Impact Analysis
-- Evaluates the joint penalty of housing and personal loans on term deposit liquidity
SELECT 
    housing,
    loan,
    COUNT(*) AS total_leads,
    ROUND(AVG(balance), 2) AS avg_balance,
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS subscribed_leads,
    ROUND((SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS conversion_pct
FROM raw_bank_marketing_ext
GROUP BY housing, loan
ORDER BY conversion_pct DESC;

-- QUERY 3: Campaign Seasonality & Month-over-Month Efficiency
-- Highlights high-volume vs high-yield months for campaign budget re-allocation
SELECT 
    month,
    COUNT(*) AS call_volume,
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS conversions,
    ROUND((SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS conversion_rate_pct,
    ROUND(AVG(duration), 1) AS avg_call_duration_sec
FROM raw_bank_marketing_ext
GROUP BY month
ORDER BY conversion_rate_pct DESC;

-- QUERY 4: Prior Campaign Outcome Engagement Compounding
-- Measures the ROI of previous relationship outcomes (success vs failure vs cold)
SELECT 
    poutcome,
    COUNT(*) AS total_contacted,
    ROUND(AVG(pdays), 1) AS avg_pdays,
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS renewed_subscriptions,
    ROUND((SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS success_rate_pct
FROM raw_bank_marketing_ext
GROUP BY poutcome
ORDER BY success_rate_pct DESC;

-- QUERY 5: Call Frequency Diminishing Returns Audit
-- Enforces data for the institutional 3-call maximum policy
SELECT 
    campaign AS contact_attempt_number,
    COUNT(*) AS leads_contacted,
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS total_subscribed,
    ROUND((SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS conversion_rate_pct
FROM raw_bank_marketing_ext
WHERE campaign <= 10
GROUP BY campaign
ORDER BY campaign ASC;

-- QUERY 6: Customer Liquidity Tiering & Asset Mobilization Potential
SELECT 
    CASE 
        WHEN balance < 0 THEN 'Negative (<€0)'
        WHEN balance >= 0 AND balance < 500 THEN 'Low (€0-€500)'
        WHEN balance >= 500 AND balance < 2000 THEN 'Moderate (€500-€2k)'
        WHEN balance >= 2000 AND balance < 10000 THEN 'High (€2k-€10k)'
        ELSE 'Affluent (>€10k)'
    END AS liquidity_tier,
    COUNT(*) AS customer_count,
    ROUND(SUM(balance) / 1000000.0, 2) AS total_aggregate_deposits_m_eur,
    SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) AS deposit_subscribers,
    ROUND((SUM(CASE WHEN y = 'yes' THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) AS tier_conversion_pct
FROM raw_bank_marketing_ext
GROUP BY 
    CASE 
        WHEN balance < 0 THEN 'Negative (<€0)'
        WHEN balance >= 0 AND balance < 500 THEN 'Low (€0-€500)'
        WHEN balance >= 500 AND balance < 2000 THEN 'Moderate (€500-€2k)'
        WHEN balance >= 2000 AND balance < 10000 THEN 'High (€2k-€10k)'
        ELSE 'Affluent (>€10k)'
    END
ORDER BY tier_conversion_pct DESC;
