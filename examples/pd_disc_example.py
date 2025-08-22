import pandas as pd
df = pd.DataFrame({
    "y_true": [0, 0, 1, 1],
    "y_pred": [0.1, 0.4, 0.35, 0.4]
})
from src.crmstudio import metrics

import importlib
importlib.reload(metrics)

auc = metrics.AUC(model_name = "pd_model", config = {"models": {"pd_model": {"metrics": [{"name": "AUC", "threshold": 0.9}]}}})
auc.compute(df["y_true"], df["y_pred"])
auc.result.to_dict()




y_pred = [1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4]
y_true = [0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,0,1]
from sklearn.metrics import roc_auc_score
import numpy as np
roc_auc_score(y_true, y_pred)

auc_curr = roc_auc_score(y_true, y_pred)
A_card = np.sum(y_true)
B_card = len(y_true) - A_card 

V10 = 


