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


cap = metrics.pd_metrics.CAPCurve("pd_model")
cap_res = cap.compute(y_true, y_pred)
fig_data = cap_res.figure_data
fig_results = cap._plot_curve(fig_data, style)

roc = metrics.pd_metrics.ROCCurve("pd_model")
roc_res = roc.compute(y_true, y_pred)
fig_data = roc_res.figure_data
fig_results = roc._plot_curve(fig_data, style)

cap.show_plot(fig_data, style)


# Method 1: Using matplotlib
import base64
from io import BytesIO
import matplotlib.pyplot as plt


pietra = metrics.pd_metrics.PietraIndex("pd_model")
pietra_res = pietra.compute(y_true, y_pred)




