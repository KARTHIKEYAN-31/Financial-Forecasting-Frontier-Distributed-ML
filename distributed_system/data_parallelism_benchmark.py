"""
============================================================================
FINANCIAL FORECASTING FRONTIER: DISTRIBUTED ML & BIG DATA MANAGEMENT
MODULE: DATA PARALLELISM & DISTRIBUTED EFFICIENCY BENCHMARKING (OPTIMIZED)
============================================================================
Author: Karthikeyan
Domain: Distributed Parallel Computing, Partition Scaling & Join Optimization
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

def heavy_compute_partition(df_chunk):
    """Simulates distributed Map-Reduce numerical feature transformations on a partition."""
    # Complex numerical operations
    arr = df_chunk['balance'].values.astype(np.float64)
    dur = df_chunk['duration'].values.astype(np.float64)
    age = df_chunk['age'].values.astype(np.float64)
    
    # Intensive polynomial feature generation and non-linear aggregations
    res = 0.0
    for _ in range(5):
        poly = np.sqrt(np.abs(arr)) * np.log1p(np.abs(dur)) / (age + 1.0)
        res += np.sum(poly)
        
    return {
        'rows': len(df_chunk),
        'poly_sum': res,
        'mean_balance': float(np.mean(arr))
    }

def benchmark_partition_parallelism(df, max_workers=multiprocessing.cpu_count()):
    """Benchmarks partition scaling across 1, 2, 4, 8 parallel partitions."""
    print("=" * 85)
    print(f"BENCHMARK 1: DATA PARALLELISM SCALING EXPERIMENT (Available CPU Cores: {max_workers})")
    print("=" * 85)
    
    scale_factor = int(250000 / len(df)) + 1
    df_large = pd.concat([df] * scale_factor, ignore_index=True).iloc[:250000]
    print(f">> Scaled Enterprise Benchmark Dataset Size: {len(df_large):,} rows | Memory: {df_large.memory_usage().sum() / 1e6:.2f} MB")
    
    partition_configs = [1, 2, 4, 8]
    results = []
    base_time = None
    
    for n_parts in partition_configs:
        chunk_size = len(df_large) // n_parts
        chunks = [df_large.iloc[i * chunk_size : (i + 1) * chunk_size] for i in range(n_parts)]
        
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_parts) as executor:
            chunk_results = list(executor.map(heavy_compute_partition, chunks))
        exec_time = time.perf_counter() - t0
        
        if n_parts == 1:
            base_time = exec_time
            speedup = 1.00
            efficiency = 100.0
        else:
            speedup = base_time / exec_time
            efficiency = (speedup / min(n_parts, max_workers)) * 100.0
            
        throughput = len(df_large) / exec_time
        
        results.append({
            'Partitions': n_parts,
            'Execution_Time_s': round(exec_time, 4),
            'Speedup_Ratio': f"{speedup:.2f}x",
            'Throughput_Rows_Sec': f"{throughput:,.0f}",
            'Parallel_Efficiency_Pct': f"{min(efficiency, 100.0):.1f}%"
        })
        print(f"Partitions: {n_parts:2d} | Latency: {exec_time:.4f}s | Speedup: {speedup:.2f}x | Throughput: {throughput:,.0f} rows/s")
        
    bench_df = pd.DataFrame(results)
    print("\n--- PARTITION SCALING BENCHMARK SUMMARY ---")
    print(bench_df.to_string(index=False))
    return bench_df

def benchmark_broadcast_vs_shuffle_join(df):
    """Benchmarks Spark Broadcast Hash Join vs Distributed Shuffle Hash Join."""
    print("\n" + "=" * 85)
    print("BENCHMARK 2: BROADCAST HASH JOIN VS DISTRIBUTED SHUFFLE JOIN")
    print("=" * 85)
    
    scale_factor = int(100000 / len(df)) + 1
    df_large = pd.concat([df] * scale_factor, ignore_index=True).iloc[:100000]
    
    job_dim = pd.DataFrame({
        'job': df['job'].unique(),
        'job_risk_tier': ['Low', 'Medium', 'Low', 'High', 'Low', 'Medium', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium'],
        'base_term_rate_pct': [3.5, 3.2, 3.8, 3.0, 3.6, 3.4, 3.1, 3.3, 4.0, 3.5, 4.2, 3.2]
    })
    
    print(f">> Fact Table Size: {len(df_large):,} rows | Dimension Table Size: {len(job_dim)} rows")
    
    # 1. Broadcast Join (Map-side direct lookup)
    t0 = time.perf_counter()
    job_map = job_dim.set_index('job').to_dict('index')
    df_bcast = df_large.copy()
    df_bcast['job_risk_tier'] = [job_map[j]['job_risk_tier'] for j in df_bcast['job']]
    df_bcast['base_term_rate_pct'] = [job_map[j]['base_term_rate_pct'] for j in df_bcast['job']]
    bcast_time = time.perf_counter() - t0
    
    # 2. Shuffle Join (Full merge hash join)
    t0 = time.perf_counter()
    df_shuffle = pd.merge(df_large, job_dim, on='job', how='inner')
    shuffle_time = time.perf_counter() - t0
    
    speedup_join = shuffle_time / bcast_time if bcast_time > 0 else 1.0
    
    print(f">> Broadcast Join Time:    {bcast_time:.4f}s (Zero Network Shuffle Overhead)")
    print(f">> Shuffle Hash Join Time: {shuffle_time:.4f}s (Full Hash Partitioning Overhead)")
    print(f">> Broadcast Speedup Advantage: {speedup_join:.2f}x faster!")
    
    join_summary = pd.DataFrame({
        'Join Strategy': ['Broadcast Hash Join (Map-Side)', 'Distributed Shuffle Hash Join'],
        'Latency (s)': [round(bcast_time, 4), round(shuffle_time, 4)],
        'Network Shuffle Overhead': ['Zero (Broadcast to Workers)', 'High (Full Cluster Shuffle)'],
        'Speedup': [f"{speedup_join:.2f}x", '1.00x (Baseline)'],
        'Best Use Case': ['Small Dimension Tables (<100MB)', 'Large Fact-to-Fact Tables']
    })
    print("\n--- DISTRIBUTED JOIN BENCHMARK SUMMARY ---")
    print(join_summary.to_string(index=False))

if __name__ == '__main__':
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'dataset.csv')
    if not os.path.exists(csv_path):
        csv_path = 'dataset/dataset.csv'
    df = pd.read_csv(csv_path)
    benchmark_partition_parallelism(df)
    benchmark_broadcast_vs_shuffle_join(df)
