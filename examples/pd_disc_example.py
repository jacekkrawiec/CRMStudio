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


"""
TODOs for tomorrow:
1. Implement DistributionAssociacionClass to the end
    a) right now I'm implementing plotting for these kind of metrics, I got stuck in a following problem -  
        contrary to curve plots, the logic here is not commong for all plots, 
        potential solution:
        move plots creation to metrics_class, central class implements only common logic, e.g. styling
2. Implement Calibration ad Stability Metrics
3. Add time/segment/subrange testing; this should follow point 1. As soon as all discrimination metrics are impletement we'll implement option to test over time and other dimensions.
4. Move to LGD


"""




