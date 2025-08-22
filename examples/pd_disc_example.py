import pandas as pd
from src.crmstudio import metrics

import importlib
importlib.reload(metrics)

y_pred = [1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4]
y_true = [0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1]

y_pred = [10*[i] for i in range(1,6)]
from itertools import chain
y_pred = list(chain.from_iterable(y_pred))
y_true = [1*(i%10 <= i/10) for i in range(len(y_pred))]


roc = metrics.pd_metrics.ROCCurve("pd_model")
roc.compute(y_true, y_pred)



import matplotlib.pyplot as plt

def plot_roc(figure_result, style_config):
    fpr = figure_result.data["fpr"]
    tpr = figure_result.data["tpr"]
    
    # Merge default style with metadata if any
    style = style_config.get("roc_curve", {})
    title = style.get("title", "ROC Curve")
    xlabel = style.get("xlabel", "False Positive Rate")
    ylabel = style.get("ylabel", "True Positive Rate")
    line_color = style.get("line_color", "blue")
    line_width = style.get("line_width", 2)
    line_style = style.get("line_style", "solid")
    
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color=line_color, linewidth=line_width, linestyle=line_style, label="ROC")
    
    # Diagonal line
    if style.get("show_diagonal", True):
        diag_color = style.get("diagonal_color", "gray")
        diag_style = style.get("diagonal_style", "dashed")
        diag_width = style.get("diagonal_width", 1)
        plt.plot([0,1], [0,1], color=diag_color, linestyle=diag_style, linewidth=diag_width, label="Random")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
