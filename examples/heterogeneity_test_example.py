"""
Example demonstrating the use of heterogeneity testing for credit risk models.

This example shows how to use the HeterogeneityTest and SubgroupCalibrationTest
classes to assess calibration consistency across different subpopulations in a 
credit risk model portfolio.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from crmstudio.metrics.pd import (
    HeterogeneityTest, SubgroupCalibrationTest, 
    AUC, ROCCurve, CalibrationCurve
)

# Create sample data with different segments
np.random.seed(42)

# Sample size
n_samples = 2000

# Create 3 segments with different calibration properties
segments = np.random.choice(['A', 'B', 'C'], size=n_samples, p=[0.5, 0.3, 0.2])

# Base probabilities for each observation
base_probs = np.random.beta(2, 5, size=n_samples)

# Adjust probabilities by segment to create heterogeneity
pd_by_segment = {
    'A': base_probs,  # Well-calibrated segment
    'B': base_probs * 0.7,  # Under-predicting segment
    'C': base_probs * 1.3   # Over-predicting segment
}

# Generate predicted probabilities based on segment
y_pred = np.array([pd_by_segment[s][i] for i, s in enumerate(segments)])

# Generate actual defaults (with intentional miscalibration in segments B and C)
def generate_defaults(probs, segment):
    if segment == 'A':
        return np.random.binomial(1, probs)  # Well-calibrated
    elif segment == 'B':
        return np.random.binomial(1, probs * 1.4)  # More defaults than predicted
    else:  # segment C
        return np.random.binomial(1, probs * 0.8)  # Fewer defaults than predicted

y_true = np.array([generate_defaults(y_pred[i], segments[i]) for i in range(n_samples)])

# Print basic statistics
print("Dataset Statistics:")
print(f"Total observations: {n_samples}")
print(f"Overall default rate: {y_true.mean():.2%}")
print(f"Average predicted PD: {y_pred.mean():.2%}")
print(f"AUC: {roc_auc_score(y_true, y_pred):.4f}")

# Statistics by segment
print("\nSegment Statistics:")
for segment in ['A', 'B', 'C']:
    mask = segments == segment
    obs = mask.sum()
    actual_dr = y_true[mask].mean()
    pred_dr = y_pred[mask].mean()
    print(f"Segment {segment}: {obs} obs, Actual DR: {actual_dr:.2%}, Predicted DR: {pred_dr:.2%}, Ratio: {actual_dr/pred_dr:.2f}")

# Run standard validation metrics
auc = AUC().compute(y_true, y_pred)
roc = ROCCurve().compute(y_true, y_pred)
cal = CalibrationCurve().compute(y_true, y_pred)

print(f"\nOverall AUC: {auc.value:.4f}")
print(f"CalibrationCurve MSE: {cal.value:.4f}")

# Run heterogeneity test
het_test = HeterogeneityTest().compute(y_true, y_pred, segments)

print("\nHeterogeneity Test Results:")
print(f"p-value: {het_test.value:.4f}")
print(f"Test passed (homogeneous calibration): {het_test.passed}")
print("\nSegment-level statistics:")
for stat in het_test.details['segment_stats']:
    print(f"Segment {stat['segment']}: Obs DR: {stat['observed_dr']:.2%}, Exp DR: {stat['expected_dr']:.2%}, " +
          f"Abs Dev: {stat['abs_deviation']:.2%}, Chi2 Contrib: {stat['chi2_contribution']:.2f}")

# Run subgroup calibration test
subgroup_test = SubgroupCalibrationTest().compute(y_true, y_pred, segments)

print("\nSubgroup Calibration Test Results:")
print(f"Proportion of acceptable subgroups: {subgroup_test.value:.2%}")
print(f"Test passed (all subgroups well-calibrated): {subgroup_test.passed}")
print("\nSubgroup-level statistics:")
for stat in subgroup_test.details['subgroup_stats']:
    print(f"Subgroup {stat['subgroup']}: Obs DR: {stat['observed_dr']:.2%}, Exp DR: {stat['expected_dr']:.2%}, " +
          f"p-value: {stat['p_value']:.4f}, Significant: {stat['statistically_significant']}")

# Plot the results
plt.figure(figsize=(15, 10))

# Plot 1: Heterogeneity Test
plt.subplot(2, 2, 1)
x = het_test.figure_data['x']
y_obs = het_test.figure_data['y']
y_exp = het_test.figure_data['y_expected']
x_indices = range(len(x))

plt.bar(x_indices, y_obs, width=0.4, label='Observed DR', alpha=0.7, color='blue')
plt.bar([i+0.4 for i in x_indices], y_exp, width=0.4, label='Expected DR', alpha=0.7, color='orange')
plt.xticks([i+0.2 for i in x_indices], x)
plt.title(het_test.figure_data['title'])
plt.xlabel(het_test.figure_data['xlabel'])
plt.ylabel(het_test.figure_data['ylabel'])
plt.legend()

# Add annotations
for i, annotation in enumerate(het_test.figure_data['annotations']):
    plt.annotate(annotation, xy=(0.05, 0.95-i*0.05), xycoords='axes fraction',
                 fontsize=9, ha='left', va='top')

# Plot 2: Subgroup Calibration Test
plt.subplot(2, 2, 2)
x = subgroup_test.figure_data['x']
y_obs = subgroup_test.figure_data['y']
y_exp = subgroup_test.figure_data['y_expected']
significant = subgroup_test.figure_data['significant']
x_indices = range(len(x))

colors = ['red' if sig else 'blue' for sig in significant]
plt.bar(x_indices, y_obs, width=0.4, label='Observed DR', alpha=0.7, color=colors)
plt.bar([i+0.4 for i in x_indices], y_exp, width=0.4, label='Expected DR', alpha=0.7, color='orange')
plt.xticks([i+0.2 for i in x_indices], x)
plt.title(subgroup_test.figure_data['title'])
plt.xlabel(subgroup_test.figure_data['xlabel'])
plt.ylabel(subgroup_test.figure_data['ylabel'])
plt.legend()

# Add annotations
for i, annotation in enumerate(subgroup_test.figure_data['annotations']):
    plt.annotate(annotation, xy=(0.05, 0.95-i*0.05), xycoords='axes fraction',
                 fontsize=9, ha='left', va='top')

# Plot 3: ROC Curve
plt.subplot(2, 2, 3)
plt.plot(roc.figure_data['x'], roc.figure_data['y'], label=f'AUC = {auc.value:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Plot 4: Calibration Curve
plt.subplot(2, 2, 4)
plt.plot(cal.figure_data['x'], cal.figure_data['y'], 'o-', label='Calibration Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title(f'Calibration Curve (MSE = {cal.value:.4f})')
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig('heterogeneity_test_example.png', dpi=300)
plt.show()

print("\nExample completed. Results saved to 'heterogeneity_test_example.png'")
