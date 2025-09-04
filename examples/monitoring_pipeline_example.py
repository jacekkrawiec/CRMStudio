"""
Example showing how to use the monitoring pipeline to run all metrics at once.

This script demonstrates:
1. Setting up a comprehensive configuration with all metrics
2. Preparing synthetic data for different metric types
3. Running the monitoring pipeline
4. Accessing and interpreting results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import yaml

# Import CRMStudio components
from crmstudio.monitoring.pipeline import MonitoringPipeline
# Import discrimination metrics
from crmstudio.metrics.pd.discrimination import AUC
# Import calibration metrics
from crmstudio.metrics.pd.calibration import HosmerLemeshow, HerfindahlIndex
# Import stability metrics
from crmstudio.metrics.pd.stability import PSI

# Set random seed for reproducibility
np.random.seed(42)

# Create directories for results and reports
os.makedirs('results', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('config', exist_ok=True)

# Define sample configuration
config = {
    "models": {
        "pd_model": {
            "metrics": [
                # Discrimination metrics
                {"name": "auc", "params": {"threshold": 0.7}},
                {"name": "roc_curve", "params": {}},
                {"name": "gini", "params": {"threshold": 0.4}},
                {"name": "ks_stat", "params": {"threshold": 0.3}},
                
                # Calibration metrics
                {"name": "hosmer_lemeshow", "params": {"n_bins": 10}},
                {"name": "calibration_curve", "params": {}},
                {"name": "brier_score", "params": {}},
                {"name": "jeffreys_test", "params": {"confidence_level": 0.95}},
                {"name": "herfindahl_index", "params": {"hi_threshold": 0.18}},
                
                # Stability metrics
                {"name": "psi", "params": {"threshold": 0.25}},
                {"name": "temporal_drift", "params": {"significance_level": 0.05}}
            ],
            "thresholds": {
                "auc": 0.7,
                "gini": 0.4,
                "ks_stat": 0.3,
                "psi": 0.25
            }
        }
    },
    "reporting": {
        "output_format": "html",
        "include_plots": True
    },
    "alerts": {
        "threshold_breach": True
    }
}

# Save config to file
config_path = "config/monitoring_config.yaml"
with open(config_path, 'w') as f:
    yaml.dump(config, f)

print(f"Configuration saved to {config_path}")

# Generate synthetic data for monitoring
print("\nGenerating synthetic data...")

# 1. Data for discrimination and calibration metrics
n_samples = 1000
# Features
X = np.random.normal(0, 1, size=(n_samples, 5))
# True labels (10% default rate)
y_true = np.random.binomial(1, 0.1, size=n_samples)
# Predicted probabilities (with some noise)
y_pred = (0.1 + 0.8 * (X[:, 0] > 0)).clip(0, 1)  # Simple model based on first feature
y_pred = np.where(y_true == 1, y_pred + 0.2, y_pred)  # Make positive examples score higher
y_pred = np.clip(y_pred + np.random.normal(0, 0.1, size=n_samples), 0, 1)  # Add noise

# 2. Data for rating-based metrics
# Create 10 rating grades (1 = best, 10 = worst)
rating_scale = list(range(1, 11))
# Assign ratings based on predicted probabilities - handle non-unique bins
try:
    # First try with duplicate handling
    ratings = pd.qcut(y_pred, 10, labels=False, duplicates='drop')
    # Map from 0-9 to our 1-10 scale
    ratings = np.array([rating_scale[int(r)] for r in ratings])
except ValueError:
    # Fallback to pd.cut with fixed bin edges if qcut still fails
    bins = np.linspace(0, 1, 11)  # 11 points create 10 bins from 0 to 1
    ratings = pd.cut(y_pred, bins=bins, labels=False)
    # Map from 0-9 to our 1-10 scale
    ratings = np.array([rating_scale[int(r)] for r in ratings])
# Exposures - higher for better ratings
exposures = np.zeros(n_samples)
for i, rating in enumerate(ratings):
    # Base exposure with some randomness
    base = 1000000 * (1 / rating)  # Better ratings get higher exposures
    variation = np.random.uniform(0.5, 1.5)
    exposures[i] = base * variation

# 3. Data for stability metrics
# PSI - Generate reference and recent distributions
reference_scores = np.random.beta(2, 5, size=800)  # Right-skewed distribution
recent_scores = np.random.beta(3, 4, size=800)  # Less skewed

# CSI - Generate reference and recent dataframes
reference_data = pd.DataFrame({
    'income': np.random.lognormal(10, 0.5, size=800),
    'age': np.random.normal(40, 10, size=800),
    'debt_ratio': np.random.beta(2, 5, size=800),
    'num_accounts': np.random.poisson(3, size=800),
    'credit_score': np.random.normal(700, 100, size=800)
})

recent_data = pd.DataFrame({
    'income': np.random.lognormal(10.3, 0.5, size=800),
    'age': np.random.normal(38, 10, size=800),
    'debt_ratio': np.random.beta(2.1, 5.1, size=800),
    'num_accounts': np.random.poisson(4, size=800),
    'credit_score': np.random.normal(705, 100, size=800)
})

# Temporal Drift - Generate time series of AUC values with a trend
end_date = datetime.now()
start_date = end_date - timedelta(days=24*30)  # Approximately 24 months
time_points = [start_date + timedelta(days=30*i) for i in range(24)]
auc_values = np.array([0.85 + (-0.004 * i) + np.random.normal(0, 0.01) for i in range(24)])
auc_values = np.clip(auc_values, 0.5, 1.0)  # Keep AUC within valid range

# Migration Analysis - Generate initial and current ratings
initial_ratings = np.random.choice(rating_scale, size=800, p=[0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.025, 0.025])
# Create current ratings with realistic migration patterns
current_ratings = np.zeros_like(initial_ratings)
for i, rating in enumerate(initial_ratings):
    # Define transition probabilities based on current rating
    if rating == 1:
        probs = [0.8, 0.15, 0.05, 0, 0, 0, 0, 0, 0, 0]
    elif rating == 2:
        probs = [0.05, 0.75, 0.15, 0.05, 0, 0, 0, 0, 0, 0]
    elif rating == 3:
        probs = [0, 0.1, 0.7, 0.15, 0.05, 0, 0, 0, 0, 0]
    elif rating == 4:
        probs = [0, 0, 0.1, 0.7, 0.15, 0.05, 0, 0, 0, 0]
    elif rating == 5:
        probs = [0, 0, 0, 0.1, 0.7, 0.15, 0.05, 0, 0, 0]
    elif rating == 6:
        probs = [0, 0, 0, 0, 0.1, 0.7, 0.15, 0.05, 0, 0]
    elif rating == 7:
        probs = [0, 0, 0, 0, 0, 0.1, 0.7, 0.15, 0.05, 0]
    elif rating == 8:
        probs = [0, 0, 0, 0, 0, 0, 0.1, 0.7, 0.15, 0.05]
    elif rating == 9:
        probs = [0, 0, 0, 0, 0, 0, 0, 0.15, 0.75, 0.1]
    else:  # rating == 10
        probs = [0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.8]
    
    # Assign current rating based on transition probabilities
    current_ratings[i] = np.random.choice(rating_scale, p=probs)

# Rating Stability - Generate time series of ratings for multiple periods
n_periods = 6
rating_time_series = []
# Initial period
initial_period_ratings = np.random.choice(rating_scale, size=800, p=[0.1, 0.2, 0.4, 0.1, 0.1, 0.05, 0.025, 0.025, 0, 0])
rating_time_series.append(initial_period_ratings)

# Generate subsequent periods with realistic rating dynamics
for period in range(1, n_periods):
    prev_ratings = rating_time_series[-1]
    new_ratings = np.zeros_like(prev_ratings)
    
    for i, prev_rating in enumerate(prev_ratings):
        # 80% chance of no change, 20% chance of moving up or down
        if np.random.random() < 0.8:
            new_ratings[i] = prev_rating
        else:
            # Movement limited to ±1 rating notch
            possible_moves = [r for r in rating_scale if abs(r - prev_rating) <= 1]
            new_ratings[i] = np.random.choice(possible_moves)
    
    rating_time_series.append(new_ratings)

# Combine all data for the pipeline
data = {
    "pd_model": {
        # Basic data for discrimination and calibration metrics
        "y_true": y_true.tolist(),  # Convert NumPy array to list
        "y_pred": y_pred.tolist(),  # Convert NumPy array to list
        "ratings": ratings.tolist(),  # Convert NumPy array to list
        "exposures": exposures.tolist(),  # Convert NumPy array to list
        
        # Stability metrics data
        "reference_scores": reference_scores.tolist(),  # Convert NumPy array to list
        "recent_scores": recent_scores.tolist(),  # Convert NumPy array to list
        "reference_data": reference_data,
        "recent_data": recent_data,
        "time_points": [t.isoformat() for t in time_points],  # Convert datetime to string
        "metric_values": auc_values.tolist(),  # Convert NumPy array to list
        "initial_ratings": initial_ratings.tolist(),  # Convert NumPy array to list
        "current_ratings": current_ratings.tolist(),  # Convert NumPy array to list
        "rating_time_series": [period.tolist() for period in rating_time_series],  # Convert list of NumPy arrays
        "rating_scale": rating_scale
    }
}

print("Data generation complete.")

# Initialize and run the monitoring pipeline
print("\nInitializing monitoring pipeline...")
pipeline = MonitoringPipeline(config_path=config_path)

print("\nRunning pipeline with all metrics...")
results = pipeline.run(data=data, save_results=True, generate_report=True)

print("\nPipeline run complete.")
print(f"Results saved to: {os.path.join('results', f'results_{pipeline.timestamp}.json')}")
print(f"Report saved to: {os.path.join('reports', f'report_{pipeline.timestamp}.html')}")

# Display summary of results
print("\nResults Summary:")
for model_name, model_results in results.items():
    print(f"\nModel: {model_name}")
    print("-" * 30)
    for metric_name, result_dict in model_results.items():
        if 'error' in result_dict:
            print(f"  {metric_name}: ERROR - {result_dict['error']}")
        else:
            result = result_dict.get('result')
            if result is None:
                continue
                
            # Format value and passed status
            value_text = f"{result.value:.4f}" if result.value is not None else "N/A"
            passed_text = "PASSED" if result.passed else "FAILED" if result.passed is not None else "N/A"
            
            print(f"  {metric_name}: {value_text} - {passed_text}")

# Display alerts
if pipeline.alerts:
    print("\nAlerts:")
    print("-" * 30)
    for alert in pipeline.alerts:
        print(f"  {alert['model']} - {alert['metric']}: {alert['value']:.4f} (threshold: {alert['threshold']:.4f})")

print("\nExample completed successfully.")
