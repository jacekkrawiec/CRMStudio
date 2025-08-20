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

- [ ] Discriminatory power: KS statistic, Gini coefficient, AUC
- [ ] Calibration metrics: Brier score, calibration plots
- [ ] Population Stability Index (PSI) for feature & score drift
- [ ] Feature-level information value (IV)
- [ ] Confusion matrix generation
- [ ] Benchmark comparison (current vs. previous runs)

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
