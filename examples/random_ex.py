import pandas as pd
import numpy as np

# Generate y_true as random binary values
y_true = np.random.randint(0, 2, size=1000)

# Generate y_pred as random scores, slightly predictive of y_true
y_pred = y_true * np.random.uniform(0.1, 0.4, size=1000) + (1 - y_true) * np.random.uniform(0.0, 0.5, size=1000)

# Assign random segments
segments = np.random.choice(['1', '2', '3'], size=1000)

# Create rating by binning y_pred into 4 groups
rating = pd.cut(y_pred, 9, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create DataFrame
df = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred,
    'segment': segments,
    'rating': rating
})

df.groupby('rating')['y_true'].sum()


from crmstudio.metrics.pd import discrimination, calibration
from crmstudio.core import base, plotting
import importlib
importlib.reload(discrimination)
importlib.reload(calibration)
importlib.reload(plotting)
importlib.reload(base)

auc = discrimination.AUC("pd_model")
auc_results = auc.compute(y_true=df['y_true'], y_pred=df['rating'])

roc = discrimination.ROCCurve("pd_model")
roc_curve = roc.compute(y_true=df['y_true'], y_pred=df['y_pred'])
roc.show_plot(roc_curve)



ece = calibration.ExpectedCalibrationError("pd_model")
ece_results = ece.compute(y_true=df['y_true'], y_pred=df['y_pred'], ratings=df['rating'])
ece.show_plot(ece_results)