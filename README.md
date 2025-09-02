# CRMStudio - IRB Model Monitoring Toolkit

CRMStudio is a Python package designed for monitoring IRB models (PD, LGD, EAD) locally within your bank infrastructure. It helps you run monitoring checks, evaluate metrics, trigger alerts, and generate reports — all **without sharing sensitive data externally**.

Attention: CRMStudio is in alpha stage, not everything is implemented yet. It is not yet available via pip, etc. For now it can be used for PD model evaluation including discrimination and calibration metrics. More features will be available soon.
---

## Quickstart

Get started with CRMStudio in just a few steps:

### 1. Install

```bash
pip install crmstudio
```

### 2. Prepare Configuration

Create a YAML config (e.g., config/monitoring_config.yaml) specifying models, metrics, and thresholds:

```yaml
models:
  pd_model:
    metrics: 
      - name: AUC
        params:
          threshold: 0.75
      - name: HosmerLemeshow
        params:
          alpha: 0.05
          n_bins: 10
    thresholds:
      psi: 0.25
      gini: 0.6
report:
  output_dir: reports/
  format: html
```

### 3. Local Data

```python
from crmstudio.data.loader import DataLoader
data = DataLoader("data/pd_data.csv").load()
```

### 4. Run Monitoring

```python
from crmstudio.monitoring.pipeline import MonitoringPipeline
MonitoringPipeline(config_path = "config/monitoring_config.yaml").run()
```

### 5. Generate Report

```python
from crmstudio.reports.generator import ReportGenerator
ReportGenerator(results_dir = 'results/', output_dir = 'results/').generate()

- [x] Done! Your model metrics are calculated, alerts triggered if thresholds are exceeded, and reports are ready.
```

## Features

### PD Model Metrics

#### Discrimination Power
- ROC Curve and AUC (Area Under the Curve)
- CAP Curve and Accuracy Ratio (AR)
- Kolmogorov-Smirnov (KS) statistic and plot
- Pietra Index (maximum separation)
- Gini coefficient with confidence intervals
- Score distributions (defaulted vs. non-defaulted)
- Lift and gain charts
- Information Value (IV)
- CIER (Conditional Information Entropy Ratio)
- AUC delta test for performance stability (ECB method)

#### Calibration Metrics
- Hosmer-Lemeshow test
- Calibration curves (reliability diagrams)
- Brier score and Brier skill score
- Expected Calibration Error (ECE)
- PD Calibration Stats (comprehensive calibration assessment)
- Binomial and Normal tests for overall calibration
- Jeffreys test for rating-based calibration

#### Heterogeneity Testing
- Heterogeneity test for calibration consistency across segments
- Subgroup calibration test for detailed subpopulation analysis
- Statistical hypothesis testing for identifying problematic segments

### Coming Soon
- Stability metrics (PSI, CSI)
- LGD and EAD/CCF metrics
- Report generation
- Monitoring pipeline

## Example

```python
from crmstudio.metrics.pd import HosmerLemeshow, HeterogeneityTest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create synthetic data
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create segments for heterogeneity testing
segments = np.random.choice(['A', 'B', 'C'], size=len(y_test))

# Train model
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]

# Calculate Hosmer-Lemeshow test
hl = HosmerLemeshow(model_name="example_model")
result = hl.compute(y_true=y_test, y_pred=y_pred)

# Display results
print(f"Hosmer-Lemeshow p-value: {result.value:.4f} ({'Passed' if result.passed else 'Failed'})")

# Test for heterogeneity across segments
het_test = HeterogeneityTest(model_name="example_model")
het_result = het_test.compute(y_true=y_test, y_pred=y_pred, segments=segments)

print(f"Heterogeneity test p-value: {het_result.value:.4f}")
print(f"Calibration homogeneity: {'Passed' if het_result.passed else 'Failed'}")

# Plot the calibration chart
hl.show_plot()

# Plot heterogeneity results
het_test.show_plot()
```

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
