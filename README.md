# CRMStudio - Credit Risk Model Studio

CRMStudio is a Python package designed for credit risk model validation, monitoring, and impact analysis. It helps you validate models, run monitoring checks, evaluate metrics, and generate reports — all **without sharing sensitive data externally**.

> **Note:** CRMStudio is currently undergoing a major restructuring to enhance usability and align with user workflows. The previous implementation has been preserved in the `obsolete` directory for reference.

## New Structure and Vision

The new structure organizes the package around validation workflows rather than technical components:

```
crmstudio/
├── validation/        # High-level API for validation tasks
│   ├── discrimination.py
│   ├── calibration.py
│   ├── stability.py
│   ├── impact.py
│   └── reporting.py
├── metrics/           # Implementations of all metrics
│   ├── discrimination/
│   ├── calibration/
│   ├── stability/
│   └── impact/
└── pipelines/         # Specialized validation workflows
    ├── irb_validation.py
    ├── annual_validation.py
    └── model_monitoring.py
```

### What's New?

The restructured CRMStudio will feature:

1. **User-Friendly API**: Intuitive functions organized around validation tasks
2. **Rich Result Objects**: Comprehensive results with built-in visualization and reporting
3. **Guided Workflows**: Specialized pipelines for common validation scenarios
4. **Regulatory Focus**: Direct support for IRB application and annual validation
5. **Business Impact**: Tools to measure the business impact of model changes

## Implementation Plan

1. **Phase 1: Core Restructuring**
   - Implement core metrics infrastructure
   - Create result objects with rich functionality
   - Develop high-level validation API

2. **Phase 2: Enhanced User Experience**
   - Implement domain-specific configuration system
   - Create specialized validation pipelines
   - Develop improved visualization and reporting

3. **Phase 3: Documentation and Examples**
   - Create comprehensive documentation
   - Develop real-world examples
   - Create Jupyter notebook tutorials

## Preview of New API

```python
# Before (old API)
auc = AUC("my_model")
gini = Gini("my_model")
ks = KSStat("my_model")

auc_result = auc.compute(y_true=defaults, y_pred=scores)
gini_result = gini.compute(y_true=defaults, y_pred=scores)
ks_result = ks.compute(y_true=defaults, y_pred=scores)

print(f"AUC: {auc_result.value:.4f}")
print(f"Gini: {gini_result.value:.4f}")
print(f"KS: {ks_result.value:.4f}")

# After (new API)
from crmstudio.validation import discrimination

# One-line call for comprehensive analysis
result = discrimination.analyze_model_power(
    y_true=defaults,
    y_pred=scores,
    segments=customer_segments
)

# Rich functionality in the result object
result.summary()
result.plot()
result.export_report("discrimination_analysis.html")
```

## Specialized Validation Pipelines

```python
from crmstudio.pipelines import IRBValidation

# Create IRB validation pipeline
irb_validator = IRBValidation(
    model_name="Mortgage PD Model",
    asset_class="mortgage",
    config=config
)

# Run comprehensive IRB validation
result = irb_validator.run(
    development_data=dev_data,
    validation_data=val_data,
    application_data=app_data
)

# Generate IRB application documentation
irb_validator.generate_documentation("irb_application_docs/")
```

## Contributing

Contributions are welcome! We are currently focused on implementing the new structure outlined above. Please see the full restructuring plan in the `documentation/restructuring.md` file for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
- Herfindahl Index for rating grade concentration

#### Heterogeneity Testing
- Heterogeneity test for calibration consistency across segments
- Subgroup calibration test for detailed subpopulation analysis
- Statistical hypothesis testing for identifying problematic segments

#### Concentration Analysis
- Herfindahl Index for rating grade concentration
- Visualization of concentration across rating grades
- Normalized concentration metrics

#### Stability Metrics
- Population Stability Index (PSI)
- Characteristic Stability Index (CSI)
- Temporal Drift Detection
- Rating Migration Analysis
- Rating Stability Analysis

### Coming Soon
- LGD and EAD/CCF metrics
- Report generation

## Monitoring Pipeline

CRMStudio provides a comprehensive monitoring pipeline that allows you to run all metrics in one go, generate reports, and trigger alerts based on thresholds.

### Pipeline Configuration

Create a YAML configuration file with all your models and metrics:

```yaml
models:
  pd_model:
    metrics:
      # Discrimination metrics
      - name: auc
        params:
          threshold: 0.7
      - name: roc_curve
        params: {}
      
      # Calibration metrics  
      - name: hosmer_lemeshow
        params:
          n_bins: 10
      - name: herfindahl_index
        params:
          hi_threshold: 0.18
      
      # Stability metrics
      - name: psi
        params:
          threshold: 0.25
    
    thresholds:
      auc: 0.7
      gini: 0.4
      psi: 0.25
      
reporting:
  output_format: html
  include_plots: true
  
alerts:
  threshold_breach: true
```

### Running the Pipeline

```python
from crmstudio.monitoring.pipeline import MonitoringPipeline

# Initialize the pipeline with your configuration
pipeline = MonitoringPipeline(config_path="config/monitoring_config.yaml")

# Prepare your data dictionary with all required data for your metrics
data = {
    "pd_model": {
        "y_true": y_true,  # True labels
        "y_pred": y_pred,  # Predicted probabilities
        "ratings": ratings,  # Rating grades
        "exposures": exposures,  # Exposures for rating-based metrics
        
        # Stability metrics data
        "reference_scores": reference_scores,
        "recent_scores": recent_scores,
        "reference_data": reference_features_df,
        "recent_data": recent_features_df,
        # Add any additional data required by your selected metrics
    }
}

# Run the pipeline
results = pipeline.run(
    data=data,
    save_results=True,  # Save results to JSON
    generate_report=True  # Generate HTML report
)

# Check for alerts
if pipeline.alerts:
    print("Alerts triggered:")
    for alert in pipeline.alerts:
        print(f"  {alert['model']} - {alert['metric']}: {alert['message']}")

# Access individual metric results
auc_result = results["pd_model"]["auc"]["result"]
print(f"AUC: {auc_result.value:.4f} - {'PASSED' if auc_result.passed else 'FAILED'}")
```

### Example Script

See the `examples/monitoring_pipeline_example.py` file for a complete example of how to use the monitoring pipeline with all metrics.

## Example

```python
from crmstudio.metrics.pd.discrimination import AUC, ROCCurve, Gini
from crmstudio.metrics.pd.calibration import HosmerLemeshow, HeterogeneityTest
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
