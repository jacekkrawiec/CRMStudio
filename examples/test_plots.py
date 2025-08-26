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
import os


#add legend to ROC by group.
auc = metrics.pd_metrics.AUC("pd_model")
results = auc.compute_over_time(y_true=df['y_true'], y_pred=df['y_pred'], time_index=df['date'], freq='M')
auc.show_group_plot(results)

auc = metrics.pd_metrics.ROCCurve("pd_model")
results = auc.compute(y_true=df['y_true'], y_pred=df['y_pred'])
auc.show_plot(results.figure_data)
results = auc.compute_by_segment(y_true=df['y_true'], y_pred=df['y_pred'], segments=df['region'])
auc.show_group_plot(results)#, filepath="C:\\Users\\jacek\\Desktop\\hist.png")

results.iloc[0,0].keys()

figure_data = results.iloc[0,0]
bin_edges = figure_data['bin_edges']

fig, ax = plt.subplots()
# Plot histograms
ax.hist(bin_edges[:-1], figure_data['hist_defaulted'],
        alpha=0.5, color= 'red',
        label='Defaulted')
ax.hist(bin_edges[:-1], figure_data['hist_non_defaulted'],
        alpha=0.5, color='green',
        label='Non-defaulted')

# Add labels and title
ax.set_xlabel(figure_data.get('xlabel', 'Score'))
ax.set_ylabel(figure_data.get('ylabel', 'Frequency'))
if group_label:
    ax.set_title(f"{group_label} (n={figure_data.get('n_obs', '')})")           

row['figure_data']

if freq not in ['M', 'Q', 'Y']:
    raise ValueError("Frequency must be one of: 'M', 'Q', 'Y'")
    
# Convert time_index to pandas datetime and period
time_index = pd.to_datetime(time_index)
period_index = pd.Series(time_index).dt.to_period(freq)

row['figure_data']

return self._compute_by_group(
    y_true=y_true,
    y_pred=y_pred,
    group_index=period_index,
    group_metadata={
        'type': 'time',
        'name': 'period'
    }
)


y_true = df['y_true'].values
y_pred = df['y_pred'].values
segments = df['region'].values

self._compute_by_group(
    y_true=np.asarray(y_true),
    y_pred=np.asarray(y_pred),
    group_index=segments,
    group_metadata={
        'type': 'segment',
        'name': 'calibration_segment',
        'labels': None
    }
)

metadata = group_metadata or {}
group_type = metadata.get('type', 'custom')
group_name = metadata.get('name', 'group')

results = []
unique_groups = np.sort(np.unique(group_index)) if group_type == 'time' else np.unique(group_index)

from crmstudio.core.base import MetricResult, FigureResult

for group in unique_groups:
    mask = group_index == group
    n_obs = np.sum(mask)
    
    if n_obs > 0:
        result = auc.compute(
            y_true=y_true[mask],
            y_pred=y_pred[mask]
        )
        if isinstance(result, MetricResult):
            results.append({
                'group': group,
                'group_type': group_type,
                'group_name': group_name,
                'n_obs': n_obs,
                'n_defaults': int(np.sum(y_true[mask])),
                'value': result.value,
                'passed': result.passed,
                **result.details
            })
        elif isinstance(result, FigureResult):
            # Handle FigureResult
            results.append({
                'group': group,
                'group_type': group_type,
                'group_name': group_name,
                'n_obs': n_obs,
                'n_defaults': int(np.sum(y_true[mask])),
                'figure_data': result.figure_data,
                **result.details
            })

result.details

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