import pandas as pd
import numpy as np
from src.crmstudio import metrics
from src.crmstudio.core import base, config_loader
from src.crmstudio.utils import helpers
import os
import yaml

CONFIG_PATH = os.path.join(os.getcwd(), "src\\crmstudio\\core\\templates", "figure_style.yaml")
style = config_loader.load_config(CONFIG_PATH)

import importlib
importlib.reload(metrics)
importlib.reload(base)

y_pred = [10*[i] for i in range(1,6)]
from itertools import chain
y_pred = list(chain.from_iterable(y_pred))
y_true = [1*(i%10 <= i/10) for i in range(len(y_pred))]


from sklearn.datasets import make_classification

y_pred, y_true = make_classification(n_samples = 1000, n_features=1, n_informative=1, n_redundant=0, n_repeated=0, n_clusters_per_class=1, class_sep=0.4)
y_pred = np.asarray(y_pred).flatten()

auc = metrics.pd_metrics.AUC("pd_model")
auc.compute(y_true = y_true, y_pred = y_pred)

roc = metrics.pd_metrics.ROCCurve("pd_model")
roc_res = roc.compute(y_true = y_true, y_pred = y_pred)

fig_data = roc_res.figure_data

roc.show_plot(fig_data, style)

ks = metrics.pd_metrics.KSStat("pd_model")
ks_res = ks.compute(fr = fig_data)
ks_res = ks.compute(y_true = y_true, y_pred = y_pred)


ks.show_plot(fig_data)

spearman = metrics.pd_metrics.SpearmansRho("pd_model")
spearman.compute(y_pred = y_pred, y_true = y_true)


iv = metrics.pd_metrics.InformationValue("pd_model")
iv.compute(y_pred = y_pred, y_true = y_true)


hist = metrics.pd_metrics.ScoreHistogram("pd_model")
ax = hist.compute(y_true = y_true, y_pred = y_pred)
hist.show_plot(ax.figure_data, style=style)


ksplot = metrics.pd_metrics.KSDistPlot("pd_model")
ax = ksplot.compute(y_true = y_true, y_pred = y_pred)
ksplot.show_plot(ax.figure_data, style=style)



n_bins = 5
y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

unique_values = np.unique(y_pred)
threshold = max(0.1 * len(y_true), 20)
is_discrete = len(unique_values) <= threshold

bad_rate = []
count = []
boxplot_stats = []
if is_discrete:
    bin_labels = np.sort(unique_values)
    bin_edges = bin_labels.tolist()
    for val in bin_labels:
        mask = y_pred == val
        y_true_bin = y_true[mask]
        count.append(np.sum(mask))
        if np.sum(mask) > 0:
            bad_rate.append(np.mean(y_true_bin))
            stats_bin = {
                "mean": float(np.mean(y_true_bin)),
                "min": float(np.min(y_true_bin)),
                "max": float(np.max(y_true_bin)),
                "q1": float(np.percentile(y_true_bin, 25)),
                "median": float(np.median(y_true_bin)),
                "q3": float(np.percentile(y_true_bin, 75))
            }
        else:
            bad_rate.append(np.nan)
            stats_bin = {
                "mean": np.nan, "min": np.nan, "max": np.nan,
                "q1": np.nan, "median": np.nan, "q3": np.nan
            }
        boxplot_stats.append(stats_bin)
else:
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_pred, quantiles)
    bin_edges = np.unique(bin_edges)
    bins_indices = np.digitize(y_pred, bin_edges, right=True)
    bin_labels = []
    for i in range(1, len(bin_edges)):
        mask = bins_indices == i
        y_true_bin = y_true[mask]
        count.append(np.sum(mask))
        if np.sum(mask) > 0:
            bad_rate.append(np.mean(y_true_bin))
            stats_bin = {
                "mean": float(np.mean(y_true_bin)),
                "whislo": float(np.min(y_true_bin)),
                "whishi": float(np.max(y_true_bin)),
                "q1": float(np.percentile(y_true_bin, 25)),
                "med": float(np.median(y_true_bin)),
                "q3": float(np.percentile(y_true_bin, 75))
            }
        else:
            bad_rate.append(np.nan)
            stats_bin = {
                "mean": np.nan, "min": np.nan, "max": np.nan,
                "q1": np.nan, "median": np.nan, "q3": np.nan
            }
        boxplot_stats.append(stats_bin)
        bin_labels.append(f"{bin_edges[i-1]:.3f} - {bin_edges[i]:.3f}")

figure_data = {
    "bin_edges": bin_edges.tolist(),
    "bad_rate": bad_rate,
    "count": count,
    "bin_labels": bin_labels if not is_discrete else bin_edges,
    "boxplot_stats": boxplot_stats
}

bin_edges = figure_data['bin_edges']
bad_rate = figure_data['bad_rate']
count = figure_data['count']
bin_labels = figure_data['bin_labels']
boxplot_stats = figure_data['boxplot_stats']

fig, ax = plt.subplots(figsize=(8,5))
bin_labels = figure_data['bin_labels']
bad_rate = figure_data['bad_rate']
count = figure_data['count']
boxplot_data = figure_data['boxplot_stats']  # Should be a list of lists/arrays, one per bin



# Add boxplots for each bin
# Position boxplots at bin centers
positions = np.arange(len(bin_labels))
box = ax.bxp(
    boxplot_data,
    positions=positions,
    widths=0.5,
    patch_artist=True,
    boxprops=dict(facecolor='gray', alpha=0.3),
    medianprops=dict(color='black'),
    showfliers=False
)

# Adjust x-ticks to match bin labels
ax.set_xticks(positions)
ax.set_xticklabels(bin_labels)

ax.set_title("Bad Rate by Score Quantile (with Boxplots)")
# Combine legends from both axes


plt.show()
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




