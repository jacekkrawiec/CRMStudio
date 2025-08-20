# CRMStudio Monitoring Tests Checklist by Model Type

## Common Tests (Applicable to PD, LGD, EAD)
### Data Quality Checks
- [ ] Missing values ratio per column
- [ ] Invalid/implausible values (e.g., negative numbers, out-of-range values)
- [ ] Constant or single-value columns
- [ ] Outlier detection (Z-score, IQR)
- [ ] Correlation drift between features
- [ ] Transition consistency checks (delinquency states monotonicity)
- [ ] Duplicate rows or IDs
- [ ] Distribution drift (univariate histograms)

### Stability & Drift Tests
- [ ] Covariate shift detection (multivariate)
- [ ] PSI over time (trend monitoring)
- [ ] Score/metric stability for similar input segments

### Threshold / Alert Checks
- [ ] Any critical metric outside defined acceptable range → alert

### Reporting Checks
- [ ] Metrics included in automated reports
- [ ] Trend charts generated correctly
- [ ] Pass/Warning/Fail flags for each metric

### Optional / Premium Advanced Tests
- [ ] Segment-level stability (e.g., by risk grade or sector)
- [ ] Stress-test sensitivity checks (shock scenarios)
- [ ] Feature importance drift (XAI-based)
- [ ] Multi-model comparison for portfolio coverage

---

## PD Model Specific Tests
### Model Performance Metrics
- [ ] Population Stability Index (PSI) for predicted PD and features
- [ ] Gini coefficient / AUC (discriminatory power)
- [ ] Kolmogorov–Smirnov (KS) statistic
- [ ] Brier score (calibration)
- [ ] Feature-level Information Value (IV)
- [ ] Confusion matrix / classification metrics (accuracy, precision, recall)
- [ ] Benchmark comparison (current vs. previous model run)

### Threshold / Alert Checks
- [ ] Gini drops below threshold → alert
- [ ] KS drops below threshold → alert

---

## LGD Model Specific Tests
### Model Performance Metrics
- [ ] Distribution comparison of predicted vs actual LGD
- [ ] Mean Absolute Error (MAE) / Root Mean Squared Error (RMSE)
- [ ] Feature-level contribution to LGD variance
- [ ] Benchmark comparison (current vs. previous model run)

### Threshold / Alert Checks
- [ ] LGD error metrics exceed threshold → alert

---

## EAD Model Specific Tests
### Model Performance Metrics
- [ ] Distribution comparison of predicted vs actual EAD
- [ ] Mean Absolute Error (MAE) / Root Mean Squared Error (RMSE)
- [ ] Feature-level contribution to EAD variance
- [ ] Benchmark comparison (current vs. previous model run)

### Threshold / Alert Checks
- [ ] EAD error metrics exceed threshold → alert
