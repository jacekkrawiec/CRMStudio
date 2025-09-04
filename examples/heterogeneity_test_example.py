"""
NOTICE: THIS EXAMPLE IS CURRENTLY DISABLED

The HeterogeneityTest and SubgroupCalibrationTest classes have been temporarily
removed and are being refactored. They will be re-implemented in a future release
with improved functionality and a more modular design.

This example will be updated once the new implementation is available.
"""

print("NOTICE: HeterogeneityTest and SubgroupCalibrationTest functionality has been temporarily removed.")
print("These classes are being refactored and will be re-implemented in a future release.")
print("Please check the documentation for updates on when this functionality will be available.")
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
