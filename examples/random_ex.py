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

n = 2000
y_true = [int(i/100*n) for i in [0.0, 0.2, 0.3, 0.4, 1.2, 2.0,3.5,4.2,8.3,12.8]]
y_pred = [int(i/100) for i in [0.05, 0.1, 0.25, 0.5, 0.75, 1.3,2,3.4,6,10]]

from crmstudio.metrics.pd import discrimination, calibration
from crmstudio.core import base, plotting
import importlib
importlib.reload(discrimination)
importlib.reload(calibration)
importlib.reload(plotting)
importlib.reload(base)

binom = calibration.NormalTest("pd_model")
binom_results = binom.compute(y_true=df['y_true'], y_pred=df['y_pred'], ratings=df['rating'])
pd.DataFrame(binom_results.details)

from scipy.stats import binom_test, binom
from scipy.special import comb
binom_test(107,162,0.250428)

ind = 0

1 - binom.cdf(y_true[ind]-1, n, y_pred[ind])

binom_test(n_def, n_tot, pred, 'greater')


n_tot = 50
n_def = 0
pred = 0.0005

res = 0
for i in range(0, n_tot+1):
    res += comb(n_tot, i) * (pred ** i) * ((1 - pred) ** (n_tot - i))
