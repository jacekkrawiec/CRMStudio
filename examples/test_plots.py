import numpy as np
import importlib
from crmstudio import metrics, core
from crmstudio.metrics.pd_metrics import PDLiftPlot, PDGainPlot, KSDistPlot, ScoreHistogram, CAPCurve, ROCCurve
from crmstudio.core.base import BaseMetric
from crmstudio.utils import helpers
import pandas as pd

# Reload modules to see the latest changes
importlib.reload(core.base)
importlib.reload(metrics.pd_metrics)
importlib.reload(metrics)

def generate_input(n_samples = 10000):
    # Set random seed for reproducibility
    np.random.seed(42)
    # Generate target variable
    y_true = np.random.binomial(1, 0.1, size=n_samples)  # 10% positives
    # Generate predicted scores (higher for positives)
    y_pred = np.where(y_true == 1,
                    np.random.beta(5, 2, size=n_samples),
                    np.random.beta(2, 5, size=n_samples))
    # Generate a numerical categorical variable (e.g., segment as int)
    categories = np.random.choice([0, 1, 2], size=n_samples)
    # Generate a continuous variable (e.g., income)
    continuous_var = np.random.normal(loc=50000, scale=15000, size=n_samples)
    # Generate a time variable: end of months from 01.01.2020 to 31.12.2024, repeated as needed
    end_of_months = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')
    time_var = np.resize(end_of_months, n_samples)
    # Generate a discrete string variable (e.g., region)
    regions = np.random.choice(['North', 'South', 'East', 'West'], size=n_samples)
    # Create DataFrame
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'category': categories,
        'income': continuous_var,
        'date': time_var,
        'region': regions
    })
    return df

df = generate_input()

auc = metrics.pd_metrics.AUC("pd_model")
results = auc.compute_by_segment(y_true=df['y_true'], y_pred=df['y_pred'], segments=df['region'])
auc.plot_group_results(results)




"""
TODOs for tomorrow:
1. Implement DistributionAssociacionClass to the end
    a) right now I'm implementing plotting for these kind of metrics, I got stuck in a following problem -  
        contrary to curve plots, the logic here is not commong for all plots, 
        potential solution:
        move plots creation to metrics_class, central class implements only common logic, e.g. styling
    b) let me first see these charts and manually look for similarities
    c) so basically what we have here is the plot of observed DR frequency vs. scores frequency, just the plot itself is different
    What they have in common are axes. So we can define 
2. Implement Calibration ad Stability Metrics
3. Add time/segment/subrange testing; this should follow point 1. As soon as all discrimination metrics are impletement we'll implement option to test over time and other dimensions.
4. Move to LGD

LS
x - score 
y - cum freq


"""