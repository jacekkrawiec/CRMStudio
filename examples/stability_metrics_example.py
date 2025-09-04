"""
Example demonstrating the use of stability metrics in CRMStudio.

This example shows how to:
1. Calculate PSI (Population Stability Index) for monitoring score distribution changes
2. Calculate CSI (Characteristic Stability Index) for monitoring input variable changes
3. Detect temporal drift in model performance metrics
4. Analyze rating migration patterns over time
5. Evaluate rating stability over multiple periods

Stability metrics are essential for ongoing monitoring of models in production
to detect when the model may need recalibration or redevelopment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import CRMStudio metrics
from crmstudio.metrics.pd import stability
from crmstudio.core import base
from crmstudio.core import plotting

import importlib
importlib.reload(plotting)
importlib.reload(base)
importlib.reload(stability)

# Set random seed for reproducibility
np.random.seed(42)

# Example 1: Population Stability Index (PSI)
print("Example 1: Population Stability Index (PSI)")
print("-" * 50)

# Generate reference distribution (e.g., from model development)
reference_scores = np.random.beta(2, 5, size=1000)  # Right-skewed distribution
print(f"Reference data: {len(reference_scores)} observations")

# Generate recent distribution with a shift
# We'll make it more centered to simulate a shift in the population
recent_scores = np.random.beta(3, 4, size=800)  # Less skewed
print(f"Recent data: {len(recent_scores)} observations")

# Calculate PSI
psi_metric = stability.PSI("example_model")
psi_result = psi_metric.compute(reference_scores=reference_scores, recent_scores=recent_scores, bins=10)

# Print results
print(f"PSI value: {psi_result.value:.4f}")
print(f"Threshold: {psi_result.threshold}")
print(f"Passed: {psi_result.passed}")
print(f"Stability category: {psi_result.details['stability_category']}")
print(f"Bin details: {len(psi_result.details['bin_details'])} bins analyzed")

# Show the plot
psi_metric.show_plot()
# Save the plot to file (create plots directory if needed)
plots_dir = 'plots'
import os
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
psi_metric.save_plot(os.path.join(plots_dir, 'psi_example.png'))

print("\nTop 3 bins contributing to PSI:")
for bin_detail in sorted(psi_result.details['bin_details'], key=lambda x: abs(x['psi_contrib']), reverse=True)[:3]:
    print(f"  Bin {bin_detail['bin']}: {bin_detail['psi_contrib']:.4f} contribution to PSI")
    print(f"  Range: [{bin_detail['lower_bound']:.2f}, {bin_detail['upper_bound']:.2f}]")
    print(f"  Reference: {bin_detail['ref_prop']:.2%}, Recent: {bin_detail['recent_prop']:.2%}")
    print()

# Example 2: Characteristic Stability Index (CSI)
print("\n\nExample 2: Characteristic Stability Index (CSI)")
print("-" * 50)

# Create reference dataframe with multiple variables
n_ref = 1000
reference_data = pd.DataFrame({
    'income': np.random.lognormal(10, 0.5, size=n_ref),
    'age': np.random.normal(40, 10, size=n_ref),
    'debt_ratio': np.random.beta(2, 5, size=n_ref),
    'num_accounts': np.random.poisson(3, size=n_ref),
    'credit_score': np.random.normal(700, 100, size=n_ref)
})
print(f"Reference data: {len(reference_data)} observations, {len(reference_data.columns)} variables")

# Create recent dataframe with shifts in some variables
n_recent = 800
recent_data = pd.DataFrame({
    # Significant shift - higher incomes
    'income': np.random.lognormal(10.3, 0.5, size=n_recent),
    # Moderate shift - slightly younger
    'age': np.random.normal(38, 10, size=n_recent),
    # No significant shift
    'debt_ratio': np.random.beta(2.1, 5.1, size=n_recent),
    # Significant shift - more accounts
    'num_accounts': np.random.poisson(4, size=n_recent),
    # No significant shift
    'credit_score': np.random.normal(705, 100, size=n_recent)
})
print(f"Recent data: {len(recent_data)} observations, {len(recent_data.columns)} variables")

# Calculate CSI
csi_metric = stability.CSI("example_model")
csi_result = csi_metric.compute(reference_data=reference_data, recent_data=recent_data, n_bins=10)

# Print results
print(f"Variables analyzed: {csi_result.details['variables_analyzed']}")
print(f"Variables exceeding threshold: {csi_result.details['variables_exceeding_threshold']}")
print("\nCSI values for each variable:")
for var_result in csi_result.details['csi_results']:
    print(f"  {var_result['variable']}: {var_result['csi']:.4f} - {var_result['stability_category']}")

# Show and save the plot
csi_metric.show_plot()
csi_metric.save_plot(os.path.join(plots_dir, 'csi_example.png'))

# Example 3: Temporal Drift Detection
print("\n\nExample 3: Temporal Drift Detection")
print("-" * 50)

# Generate dates for the last 24 months
end_date = datetime.now()
start_date = end_date - timedelta(days=24*30)  # Approximately 24 months
dates = [start_date + timedelta(days=30*i) for i in range(24)]

# Generate AUC values with a decreasing trend
# Start around 0.85 and gradually decrease to 0.75
base_auc = 0.85
trend_slope = -0.004  # Decreasing trend
noise_level = 0.01

auc_values = np.array([base_auc + trend_slope * i + np.random.normal(0, noise_level) for i in range(24)])
auc_values = np.clip(auc_values, 0.5, 1.0)  # Keep AUC within valid range

# Print the data
print(f"Time points: {len(dates)} months")
print(f"Starting AUC: {auc_values[0]:.4f}")
print(f"Ending AUC: {auc_values[-1]:.4f}")

# Detect drift
drift_metric = stability.TemporalDriftDetection("example_model")
drift_result = drift_metric.compute(time_points=np.array(dates), metric_values=auc_values, metric_name="AUC")

# Print results
print(f"\nTrend detected: {drift_result.details['trend_results']['trend_detected']}")
print(f"Trend direction: {drift_result.details['trend_results']['trend_direction']}")
print(f"p-value: {drift_result.details['trend_results']['p_value']:.4f}")
print(f"Concerning drift: {drift_result.details['trend_results']['concerning_drift']}")
print(f"Slope: {drift_result.details['trend_results']['slope']:.6f} per month")
print(f"Next period prediction: {drift_result.details['trend_results']['future_prediction']:.4f}")

# Show and save the plot
drift_metric.show_plot()
drift_metric.save_plot(os.path.join(plots_dir, 'drift_example.png'))

# Example 4: Migration Analysis
print("\n\nExample 4: Migration Analysis")
print("-" * 50)

# Generate synthetic rating data for two periods
n_obligors = 1000
rating_scale = [1, 2, 3, 4, 5]  # 1 = best, 5 = worst

# Generate initial ratings
initial_rating_probs = [0.1, 0.2, 0.4, 0.2, 0.1]  # Probability distribution across ratings
initial_ratings = np.random.choice(rating_scale, size=n_obligors, p=initial_rating_probs)

# Generate current ratings with realistic migration patterns
# Most stay the same, some move up or down one notch, few move two notches
current_ratings = np.zeros_like(initial_ratings)

for i, rating in enumerate(initial_ratings):
    # Define probabilities for each transition based on current rating
    if rating == 1:
        # Best rating can only stay the same or get worse
        probs = [0.8, 0.15, 0.05, 0, 0]
    elif rating == 2:
        probs = [0.05, 0.75, 0.15, 0.05, 0]
    elif rating == 3:
        probs = [0, 0.1, 0.7, 0.15, 0.05]
    elif rating == 4:
        probs = [0, 0.05, 0.15, 0.75, 0.05]
    else:  # rating == 5
        probs = [0, 0, 0.05, 0.15, 0.8]
    
    current_ratings[i] = np.random.choice(rating_scale, p=probs)

# Calculate migration matrix
migration_metric = stability.MigrationAnalysis("example_model")
migration_result = migration_metric.compute(
    initial_ratings=initial_ratings, 
    current_ratings=current_ratings,
    rating_scale=rating_scale
)

# Print results
print(f"Total obligors: {migration_result.details['total_obligors']}")
print(f"Stability ratio: {migration_result.details['stability_ratio']:.2f}")
print(f"Upgrade ratio: {migration_result.details['upgrade_ratio']:.2f}")
print(f"Downgrade ratio: {migration_result.details['downgrade_ratio']:.2f}")
print(f"Mobility index: {migration_result.details['mobility_shorrocks']:.2f}")

print("\nMigration probability matrix:")
prob_matrix = np.array(migration_result.details['probability_matrix'])
df_matrix = pd.DataFrame(
    prob_matrix, 
    index=[f"From {r}" for r in rating_scale],
    columns=[f"To {r}" for r in rating_scale]
)
print(df_matrix.round(2))

# Show and save the plot
migration_metric.show_plot()
migration_metric.save_plot(os.path.join(plots_dir, 'migration_example.png'))

# Example 5: Rating Stability Analysis
print("\n\nExample 5: Rating Stability Analysis")
print("-" * 50)

# Generate synthetic rating data for multiple periods
n_periods = 6
n_obligors = 800

# Generate time series of ratings for each obligor
# Most ratings should be stable with occasional changes
rating_time_series = []

# Initial period - use distribution skewed toward middle ratings
initial_ratings = np.random.choice(rating_scale, size=n_obligors, p=[0.1, 0.2, 0.4, 0.2, 0.1])
rating_time_series.append(initial_ratings)

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

# Calculate rating stability
stability_metric = stability.RatingStabilityAnalysis("example_model")
stability_result = stability_metric.compute(
    rating_time_series=rating_time_series,
    time_points=[f"Period {i+1}" for i in range(n_periods)]
)

# Print results
print(f"Total obligors: {stability_result.details['n_obligors']}")
print(f"Number of periods: {stability_result.details['n_periods']}")
print(f"Overall change ratio: {stability_result.details['overall_change_ratio']:.2f}")
print(f"Mean change ratio per obligor: {stability_result.details['mean_change_ratio']:.2f}")
print(f"Reversal ratio: {stability_result.details['reversal_ratio']:.2f}")

print("\nObligor statistics:")
print(f"  Obligors with no rating changes: {stability_result.details['obligor_stats_summary']['n_obligors_with_no_changes']}")
print(f"  Obligors with one rating change: {stability_result.details['obligor_stats_summary']['n_obligors_with_one_change']}")
print(f"  Obligors with multiple rating changes: {stability_result.details['obligor_stats_summary']['n_obligors_with_multiple_changes']}")

print("\nPeriod-by-period change rates:")
for i, rate in enumerate(stability_result.details['period_change_rates']):
    print(f"  Period {i+1} to {i+2}: {rate:.2f}")

# Show and save the plot
stability_metric.show_plot()
stability_metric.save_plot(os.path.join(plots_dir, 'rating_stability_example.png'))

print("\nExample completed successfully.")
