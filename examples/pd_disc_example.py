import pandas as pd
from src.crmstudio import metrics
from src.crmstudio.core import base
from src.crmstudio.utils import helpers


import importlib
importlib.reload(metrics)
importlib.reload(base)

y_pred = [1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4]
y_true = [0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1]

y_pred = [10*[i] for i in range(1,6)]
from itertools import chain
y_pred = list(chain.from_iterable(y_pred))
y_true = [1*(i%10 <= i/10) for i in range(len(y_pred))]


roc = metrics.pd_metrics.ROCCurve("pd_model")
res = roc.compute(y_true, y_pred)
res.figure_data


import matplotlib.pyplot as plt
import os

STYLE_PATH = os.path.join(os.getcwd(), "src\\crmstudio\\core\\templates", "figure_style.yaml")
from src.crmstudio.core.config_loader import load_config
style_config = load_config(STYLE_PATH)







import yaml
import matplotlib.pyplot as plt

# --- Load Style ---
def load_plot_style(path="plot_style.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# --- Apply Style ---
def apply_plot_style(style):
    plt.rcParams.update({
        "font.family": style["fonts"]["family"],
        "font.size": style["fonts"]["size"],
        "axes.titlesize": style["fonts"]["title_size"],
        "axes.labelsize": style["fonts"]["label_size"],
        "xtick.labelsize": style["fonts"]["tick_size"],
        "ytick.labelsize": style["fonts"]["tick_size"],
        "legend.fontsize": style["fonts"]["legend_size"],
        "figure.dpi": style["figure"]["dpi"],
        "figure.figsize": style["figure"]["figsize"],
        "axes.facecolor": style["axes"]["facecolor"],
        "axes.edgecolor": style["axes"]["edgecolor"],
        "axes.grid": style["axes"]["grid"],
        "grid.color": style["colors"]["grid"],
        "grid.linestyle": style["axes"]["grid_linestyle"],
        "grid.linewidth": style["axes"]["grid_linewidth"],
        "grid.alpha": style["axes"]["grid_alpha"],
        "axes.spines.top": style["axes"]["spine_visibility"],
        "axes.spines.right": style["axes"]["spine_visibility"],
        "axes.linewidth": style["axes"]["linewidth"],
    })
import numpy as np
# --- Example ROC Plot ---
def plot_roc(fpr, tpr, auc_score, style):
    apply_plot_style(style)
    palette = style["colors"]["palette"]
    
    fig, ax = plt.subplots()
    for i in range(5):
        ax.plot([max(x,0) for x in np.asarray(fpr) - i/100], tpr, color=palette[i], lw=2, label=f"ROC curve (AUC = {auc_score:.2f})")
    
    
    if style["roc_curve"]["show_diagonal"]:
        ax.plot([0, 1], [0, 1], linestyle=style["roc_curve"]["diagonal_linestyle"],
                linewidth=style["roc_curve"]["diagonal_linewidth"],
                alpha=style["roc_curve"]["diagonal_alpha"],
                color=style["colors"]["diagonal_line"])
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc=style["legend"]["loc"], frameon=style["legend"]["frameon"])
    fig.patch.set_facecolor(style["figure"]["background_color"])
    plt.show()

# Usage:
fpr = res.figure_data.get("fpr")
tpr = res.figure_data.get("tpr")
auc_score = 0.6
style = style_config
plot_roc(fpr, tpr, auc_score, style_config)
