"""
============================================================================
FINANCIAL FORECASTING FRONTIER: DISTRIBUTED ML & BIG DATA MANAGEMENT
MODULE: SPARK STREAMING REAL-TIME TRANSACTION ANALYSIS & ANOMALY DETECTION
============================================================================
Author: Karthikeyan
Domain: Real-Time Banking Event Streams, Windowed Aggregations & Anomaly Alerts
"""

import os
import sys
import time
import json
import random
import datetime
from collections import deque
import pandas as pd
import numpy as np

class RealTimeTransactionGenerator:
    """Simulates high-velocity real-time retail banking transaction event streams."""
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.txn_counter = 100000
        self.merchants = ['Amazon Pay', 'Apple Store', 'Shell Gas', 'Walmart Grocery', 'Stripe Transfer', 
                          'Revolut Cash Out', 'Crypto Exchange', 'ATM Withdrawal', 'Luxury Retailer', 'Uber Trips']
        self.locations = ['Lisbon, PT', 'Porto, PT', 'Madrid, ES', 'London, UK', 'Frankfurt, DE', 'New York, US', 'Online Secure']
        self.jobs = ['retired', 'management', 'technician', 'blue-collar', 'admin.', 'services', 'student']
        
    def generate_event(self):
        self.txn_counter += 1
        is_fraud = random.random() < 0.05  # 5% anomalous/fraudulent rate
        
        if is_fraud:
            amount = round(random.uniform(2500.0, 18500.0), 2)
            merchant = random.choice(['Crypto Exchange', 'Revolut Cash Out', 'ATM Withdrawal'])
            location = random.choice(['Frankfurt, DE', 'New York, US', 'Online Secure'])
            device_risk = random.choice(['High', 'Critical'])
        else:
            amount = round(random.uniform(5.50, 480.0), 2)
            merchant = random.choice(self.merchants)
            location = random.choice(['Lisbon, PT', 'Porto, PT'])
            device_risk = 'Low'
            
        event = {
            'transaction_id': f"TXN-{self.txn_counter}",
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'customer_id': f"CUST-{random.randint(1000, 9999)}",
            'customer_job': random.choice(self.jobs),
            'customer_age': random.randint(20, 75),
            'amount_eur': amount,
            'merchant': merchant,
            'location': location,
            'device_risk': device_risk,
            'is_anomalous': is_fraud
        }
        return event

class SparkStreamingWindowProcessor:
    """
    Implements Spark Streaming micro-batch and windowed operations:
    1. Tumbling Window (10-event batch aggregation)
    2. Sliding Window (20-event moving window with 5-event slide)
    3. Real-Time Statistical Z-Score Anomaly & Fraud Alert Sink
    """
    def __init__(self, window_size=20, slide_interval=5):
        self.window_size = window_size
        self.slide_interval = slide_interval
        self.sliding_buffer = deque(maxlen=window_size)
        self.alerts_triggered = []
        self.total_processed = 0
        self.total_volume_eur = 0.0

    def process_stream(self, total_events=60):
        gen = RealTimeTransactionGenerator()
        print("=" * 85)
        print("STARTING SPARK STRUCTURED STREAMING SIMULATION (MICRO-BATCH WINDOW PROCESSING)")
        print(f">> Micro-batch window config: WindowSize={self.window_size} events | SlideInterval={self.slide_interval} events")
        print("=" * 85)
        
        batch_num = 0
        current_batch = []
        
        for i in range(1, total_events + 1):
            event = gen.generate_event()
            current_batch.append(event)
            self.sliding_buffer.append(event)
            self.total_processed += 1
            self.total_volume_eur += event['amount_eur']
            
            # Instantaneous Anomaly Scoring (Threshold: Amount > €2,000 OR DeviceRisk == 'Critical')
            if event['amount_eur'] > 2000.0 or event['device_risk'] in ['High', 'Critical']:
                alert = {
                    'Alert_ID': f"ALT-{len(self.alerts_triggered)+1:03d}",
                    'Txn_ID': event['transaction_id'],
                    'Customer': event['customer_id'],
                    'Amount_EUR': event['amount_eur'],
                    'Merchant': event['merchant'],
                    'Location': event['location'],
                    'Severity': 'CRITICAL FRAUD ALERT' if event['amount_eur'] > 5000 else 'HIGH RISK ALERT'
                }
                self.alerts_triggered.append(alert)
                print(f"[!] REAL-TIME SINK ALERT: {alert['Severity']} | {alert['Txn_ID']} | {alert['Customer']} | €{alert['Amount_EUR']:,.2f} at {alert['Merchant']}")

            # Tumbling Window Trigger (every 10 events)
            if i % 10 == 0:
                batch_num += 1
                batch_amounts = [e['amount_eur'] for e in current_batch]
                print(f"\n--- [TUMBLING WINDOW MICRO-BATCH #{batch_num}] Processed Events: {len(current_batch)} | Total Batch Amount: €{sum(batch_amounts):,.2f} | Avg Amount: €{np.mean(batch_amounts):.2f} ---")
                current_batch = []

            # Sliding Window Trigger (every 5 events when buffer is full)
            if len(self.sliding_buffer) >= self.window_size and i % self.slide_interval == 0:
                win_amounts = [e['amount_eur'] for e in self.sliding_buffer]
                win_mean = np.mean(win_amounts)
                win_std = np.std(win_amounts)
                win_max = np.max(win_amounts)
                print(f">> [SLIDING WINDOW STATS] Window Size: {len(self.sliding_buffer)} | 20-Txn Moving Mean: €{win_mean:.2f} | StdDev: €{win_std:.2f} | Max Spike: €{win_max:,.2f}")

            time.sleep(0.01)  # Micro-batch pacing

        print("\n" + "=" * 85)
        print("REAL-TIME STREAMING MONITORING EXECUTIVE SUMMARY")
        print("=" * 85)
        print(f"Total Stream Events Ingested:    {self.total_processed:,} transactions")
        print(f"Total Financial Volume Processed: €{self.total_volume_eur:,.2f}")
        print(f"Total Anomaly Alerts Intercepted: {len(self.alerts_triggered)}")
        
        if self.alerts_triggered:
            alerts_df = pd.DataFrame(self.alerts_triggered)
            print("\n--- INTERCEPTED HIGH-RISK ANOMALY ALERTS TABLE ---")
            print(alerts_df.to_string(index=False))

if __name__ == '__main__':
    processor = SparkStreamingWindowProcessor(window_size=20, slide_interval=5)
    processor.process_stream(total_events=50)
