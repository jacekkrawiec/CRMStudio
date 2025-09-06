# IRB-Tools Feature Checklist

## 1. Data Input & Management

- [ ] Load credit risk data from CSV, Parquet, Excel
- [ ] Schema validation (expected columns, types)
- [ ] Metadata management (model ID, run timestamp, dataset version)
- [ ] Synthetic dataset generator for testing & demos
- [ ] Local caching of historical runs for trend analysis

## 2. Data Quality Profiler

- [ ] Missing value detection & ratios per column
- [ ] Implausible value detection (negative EAD, LGD > 1, PD < 0)
- [ ] Constant or unique value detection
- [ ] Outlier detection (z-score, IQR)
- [ ] Correlation checks (feature dependency validation)
- [ ] Transition consistency checks (delinquency state monotonicity)
- [ ] HTML/PDF summary reports

## 3. Model Performance Metrics

### 3.1 PD Model Metrics

#### 3.1.1 Discriminatory Power
- [x] ROC Curve and AUC (Area Under the Curve)
- [x] CAP Curve and Accuracy Ratio (AR)
- [x] Kolmogorov-Smirnov (KS) statistic
- [x] Pietra Index (maximum separation)
- [x] Gini coefficient with confidence intervals
- [x] Score distributions (defaulted vs. non-defaulted)
- [x] Lift and gain charts
- [x] Information Value (IV)
- [x] CIER (Conditional Information Entropy Ratio)
- [x] AUC delta test for performance stability (ECB method)

#### 3.1.2 Calibration Metrics
- [x] Hosmer-Lemeshow test
- [x] Calibration curves (reliability diagrams)
- [x] Brier score and Brier skill score
- [x] Expected Calibration Error (ECE)
- [x] PD Calibration Stats (comprehensive calibration assessment)

#### 3.1.3 Stability Metrics
- [ ] Population Stability Index (PSI)
- [ ] Characteristic Stability Index (CSI)
- [ ] Distribution drift detection
- [ ] PD model performance tracking over time
- [ ] Simple out-of-time validation framework

### 3.2 LGD Model Metrics (Planned)
- [ ] MSE, MAE, RMSE for LGD predictions
- [ ] R-squared and adjusted R-squared
- [ ] Calibration assessment (predicted vs. observed LGD)
- [ ] ROC AUC for binary component (zero vs. non-zero LGD)
- [ ] Segmented performance metrics (by collateral type, exposure size, etc.)

### 3.3 EAD/CCF Model Metrics (Planned)
- [ ] Conversion Factor accuracy metrics
- [ ] EAD over/underestimation analysis
- [ ] Segmented performance by product type and facility characteristics
- [ ] Stability analysis for conversion factors

## 4. Model Monitoring & Validation

- [ ] Config-driven monitoring pipelines (YAML/JSON)
- [ ] Threshold-based alerts for KPI drift
- [ ] Multi-period trend analysis (time series of metrics)
- [ ] Historical run comparison and plots
- [ ] CLI and Python API for automation
- [ ] Custom validator integration (extensible plugin system)

## 5. Reporting Module

- [ ] Generate HTML / PDF reports with charts & tables
- [ ] LaTeX / Word export for regulatory submissions
- [ ] Include data quality, model performance, drift analysis in one report
- [ ] Configurable report templates (customizable branding & layout)
- [ ] Automated summary section with “Pass/Warning/Fail” flags

## 6. Explainability & Governance (Optional / Premium)

- [ ] Model explainability: SHAP, partial dependence plots, monotonicity checks
- [ ] Metadata logging for governance & audit trail
- [ ] Auto-generated governance documentation (assumptions, limitations)
- [ ] Versioning of models and datasets for traceability

## 7. Regulatory / Scenario Modules (Optional / Premium)

- [ ] Pre-coded Basel IV / EBA compliance checks
- [ ] Stress testing / scenario impact analysis on PD, LGD, EAD
- [ ] Synthetic scenario simulation (macro shocks, sectoral impacts)
- [ ] IFRS9 vs IRB comparison module

## 8. Visualization & Dashboards

- [ ] Interactive charts with Plotly or Matplotlib
- [ ] Multi-metric dashboard for monitoring multiple models
- [ ] Exportable figures for reports or presentations
- [ ] Heatmaps for feature drift and correlations

## 9. Extensibility & Integration

- [ ] Python API for integration into existing model pipelines
- [ ] Plugin system for custom validators or metrics
- [ ] Option to schedule recurring monitoring runs (cron/airflow)
- [ ] Local database support for storing past runs (SQLite or Parquet)

## 10. Open-Source / Enterprise Layers

- [ ] Open-Source core: basic data quality checks, model metrics, simple report export, synthetic dataset generator
- [ ] Premium: regulatory-ready templates, interactive dashboards, advanced metrics, multi-period trends, scenario simulations
- [ ] Consulting/Service: training workshops, model validation support, customization for bank-specific pipelines
