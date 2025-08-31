"""
Example demonstrating the simplified plotting approach in CRMStudio.

This example shows how to use the simplified plotting functionality where:
1. BaseMetric.show_plot() and BaseMetric.save_plot() are the only entry points for plotting
2. These methods automatically handle both individual and group analysis results
3. The PlottingService handles the appropriate visualization based on the data type
"""

import numpy as np
import importlib
import sys
import pandas as pd
import matplotlib.pyplot as plt

# First import the modules (to make them available for reloading)
from crmstudio import core, metrics, utils

# Full reload sequence - order matters!
# First reload lower level modules, then the modules that depend on them
importlib.reload(core.data_classes)
importlib.reload(core.config_loader)
importlib.reload(utils.helpers)
importlib.reload(core.plotting)
importlib.reload(core.base)
importlib.reload(core)
importlib.reload(metrics.pd_metrics) 
importlib.reload(metrics)

# Force Python to reload modules from disk by removing them from sys.modules cache
# This is a more aggressive approach if the above reloads don't work
for key in list(sys.modules.keys()):
    if key.startswith('crmstudio'):
        sys.modules.pop(key, None)

# Re-import after clearing cache
from crmstudio import metrics
from crmstudio import core

def generate_input(n_samples=10000):
    """Generate synthetic data for testing"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate target variable
    y_true = np.random.binomial(1, 0.5, size=n_samples)  # 50% positives

    # Generate predicted scores (higher for positives)
    y_pred = np.where(y_true == 1,
                      np.random.beta(5, 2, size=n_samples),
                      np.random.beta(2, 5, size=n_samples))
    
    # Generate a time variable: end of months from 01.01.2020 to 31.12.2022, repeated as needed
    end_of_months = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')
    time_var = np.resize(end_of_months, n_samples)
    
    # Generate a discrete string variable (e.g., region)
    regions = np.random.choice(['North', 'South', 'East', 'West'], size=n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'date': time_var,
        'region': regions
    })
    return df

print("Generating test data...")
df = generate_input()

#Testing AUC with threshold

config = {"models": {"test_model": {"metrics": [{
                    "name": "auc",
                    "threshold": 0.8
                }
            ]
        }
    }
}


auc = metrics.pd_metrics.AUC("test_model", config = config)
auc.config
result = auc.compute(y_true=df['y_true'], y_pred=df['y_pred'])
result.passed

assert result.threshold == 0.8
assert result.passed is True  # 1.0 >= 0.8

# Example 1: Individual metric with show_plot
print("\nExample 1: Individual metric with show_plot")
roc = metrics.pd_metrics.ROCCurve("pd_model")
result = roc.compute(y_true=df['y_true'], y_pred=df['y_pred'])
print(f"ROC AUC: {result.value:.4f}")
# Calling show_plot with no arguments uses stored result
roc.show_plot(result)


result.figure_data


# Example 2: Distribution metric with show_plot and result passed
# EMPTY

# Example 3: Group analysis (by segment) with show_plot
print("\nExample 3: Group analysis (by segment) with show_plot")
roc_segment_results = roc.compute_by_segment(y_true=df['y_true'], y_pred=df['y_pred'], segments=df['region'])
# Using the same show_plot method for group results
roc.show_plot(roc_segment_results)

# Example 4: Group analysis (over time) with show_plot
print("\nExample 4: Group analysis (over time) with show_plot")
roc_time_results = roc.compute_over_time(y_true=df['y_true'], y_pred=df['y_pred'], time_index=df['date'], freq='Q')
# Using the same show_plot method for time-based results
roc.show_plot(roc_time_results)

# Example 5: Scalar metric (AUC) by segment with show_plot
print("\nExample 5: Scalar metric (AUC) by segment with show_plot")
auc = metrics.pd_metrics.AUC("pd_model")
auc_segment_results = auc.compute_by_segment(y_true=df['y_true'], y_pred=df['y_pred'], segments=df['region'])
# Using the same show_plot method for scalar metric group results
auc.show_plot(auc_segment_results)

# Example 6: plot single scalar value
auc = metrics.pd_metrics.AUC("pd_model")
result = auc.compute(y_true=df['y_true'], y_pred=df['y_pred'])
auc.show_plot(result)

# Example 7: plot single scalar value
auc = metrics.pd_metrics.SpearmansRho("pd_model")
result = auc.compute(y_true=df['y_true'], y_pred=df['y_pred'])
auc.show_plot(result)

# Example 8: plot single scalar value
auc = metrics.pd_metrics.SpearmansRho("pd_model")
result = auc.compute(y_true=df['y_true'], y_pred=df['y_pred'])
auc.show_plot(result)

# Example 9: plot ScoreHistogram
histogram = metrics.pd_metrics.ScoreHistogram("pd_model")
result = histogram.compute(y_true=df['y_true'], y_pred=df['y_pred'])
histogram.show_plot(result)

# Example 10: plot ScoreHistogram by Segment
histogram = metrics.pd_metrics.ScoreHistogram("pd_model")
result = histogram.compute_by_segment(y_true=df['y_true'], y_pred=df['y_pred'], segments=df['region'])
histogram.show_plot(result)

# Example 11: plot ScoreHistogram over Time
histogram = metrics.pd_metrics.ScoreHistogram("pd_model")
result = histogram.compute_over_time(y_true=df['y_true'], y_pred=df['y_pred'], time_index=df['date'], freq='M')
histogram.show_plot(result)


# Example 12: plot KSDistPlot
ks_dist_plot = metrics.pd_metrics.KSDistPlot("pd_model")
result = ks_dist_plot.compute(y_true=df['y_true'], y_pred=df['y_pred'])
ks_dist_plot.show_plot(result)

# Example 13: plot KSDistPlot by Segment
ks_dist_plot = metrics.pd_metrics.KSDistPlot("pd_model")
result = ks_dist_plot.compute_by_segment(y_true=df['y_true'], y_pred=df['y_pred'], segments=df['region'])
ks_dist_plot.show_plot(result)

# Example 14: plot KSDistPlot over Time
ks_dist_plot = metrics.pd_metrics.KSDistPlot("pd_model")
result = ks_dist_plot.compute_over_time(y_true=df['y_true'], y_pred=df['y_pred'], time_index=df['date'], freq='M')
ks_dist_plot.show_plot(result)

# Example 15: plot PDGainPlot
pd_gain_plot = metrics.pd_metrics.PDGainPlot("pd_model")
result = pd_gain_plot.compute(y_true=df['y_true'], y_pred=df['y_pred'])
pd_gain_plot.show_plot(result)

# Example 16: plot PDGainPlot by Segment
pd_gain_plot = metrics.pd_metrics.PDGainPlot("pd_model")
result = pd_gain_plot.compute_by_segment(y_true=df['y_true'], y_pred=df['y_pred'], segments=df['region'])
pd_gain_plot.show_plot(result)

# Example 17: plot PDGainPlot over Time
pd_gain_plot = metrics.pd_metrics.PDGainPlot("pd_model")
result = pd_gain_plot.compute_over_time(y_true=df['y_true'], y_pred=df['y_pred'], time_index=df['date'], freq='M')
pd_gain_plot.show_plot(result)

