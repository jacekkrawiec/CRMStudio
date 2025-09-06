"""
Example demonstrating the use of RatingHeterogeneityTest for credit risk models.

This example shows how to use the RatingHeterogeneityTest class to assess whether
consecutive rating grades show statistically significant differences in observed
default rates, which is essential for a well-differentiated rating system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from crmstudio.metrics.pd import calibration
import importlib
importlib.reload(calibration)
# Create sample data with rating grades
np.random.seed(42)

# Sample size
n_samples = 3000

# Create synthetic rating grades (1 = best, 5 = worst)
# Distribution skewed towards better ratings
rating_probs = [0.3, 0.25, 0.2, 0.15, 0.1]
ratings = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=rating_probs)

# Define default rates for each rating grade
# Well-differentiated default rates with clear separation
target_dr_by_rating = {
    1: 0.01,  # 1% default rate for rating 1
    2: 0.03,  # 3% default rate for rating 2
    3: 0.08,  # 8% default rate for rating 3
    4: 0.15,  # 15% default rate for rating 4
    5: 0.30   # 30% default rate for rating 5
}

# Generate actual defaults based on rating-specific default rates
y_true = np.array([np.random.binomial(1, target_dr_by_rating[r]) for r in ratings])

# Generate predicted PDs close to the actual default rates with some noise
# (adding noise to represent model estimation error)
noise_factor = 0.2  # 20% noise
y_pred = np.array([target_dr_by_rating[r] * (1 + noise_factor * (np.random.random() - 0.5)) 
                   for r in ratings])

# Ensure predicted PDs are between 0 and 1
y_pred = np.clip(y_pred, 0.001, 0.999)

# Print basic statistics
print("Dataset Statistics:")
print(f"Total observations: {n_samples}")
print(f"Overall default rate: {y_true.mean():.2%}")
print(f"Average predicted PD: {y_pred.mean():.2%}")

# Print statistics by rating grade
print("\nRating Grade Statistics:")
df = pd.DataFrame({'rating': ratings, 'actual': y_true, 'predicted': y_pred})
rating_stats = df.groupby('rating').agg({
    'actual': ['count', 'mean'],
    'predicted': 'mean'
}).reset_index()
rating_stats.columns = ['Rating', 'Count', 'Observed DR', 'Predicted PD']
print(rating_stats.to_string(index=False, float_format=lambda x: f"{x:.2%}" if x < 1 else f"{int(x)}"))

# Run calibration curve analysis
cal = calibration.CalibrationCurve("Example Model").compute(y_true = y_true, y_pred = y_pred, ratings=ratings)
print(f"\nCalibration Error: {cal.value:.4f}")

# Run rating heterogeneity test with Fisher's exact test
het_test_fisher = calibration.RatingHeterogeneityTest("Example Model").compute(
    y_true = y_true, y_pred = y_pred, ratings=ratings, test_method="fisher"
)

# Run rating heterogeneity test with proportion z-test
het_test_prop = calibration.RatingHeterogeneityTest("Example Model").compute(
    y_true = y_true, y_pred = y_pred, ratings=ratings, test_method="proportion"
)

# Print heterogeneity test results
print("\nRating Heterogeneity Test Results (Fisher's exact test):")
print(f"Proportion of significant pairs: {het_test_fisher.value:.2%}")
print(f"Test passed (all adjacent ratings differentiated): {het_test_fisher.passed}")

print("\nPairwise comparison details (Fisher's exact test):")
for pair in het_test_fisher.details['pair_results']:
    print(f"Ratings {pair['rating1']} vs {pair['rating2']}: " +
          f"DR {pair['dr1']:.2%} vs {pair['dr2']:.2%}, " +
          f"p-value: {pair['p_value']:.4f}, " +
          f"Significant: {pair['is_significant']}")

# Create a custom plot to visualize the results
plt.figure(figsize=(12, 8))

# Plot 1: Default rates by rating with significance indicators
plt.subplot(2, 1, 1)
x = range(len(het_test_fisher.figure_data['x']))
y = het_test_fisher.figure_data['y']

# Plot bars for default rates
bars = plt.bar(x, y, alpha=0.7)

# Add sample size as text on top of each bar
for i, (count, bar) in enumerate(zip(het_test_fisher.figure_data['n_obs'], bars)):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
             f'n={count}', ha='center', va='bottom', fontsize=9)

# Add significance indicators between bars
for i in range(len(x) - 1):
    if het_test_fisher.figure_data['significant_markers'][i]:
        # Significant difference
        plt.plot([x[i] + 0.4, x[i+1] - 0.4], [max(y[i], y[i+1]) + 0.02, max(y[i], y[i+1]) + 0.02], 
                 'g-', linewidth=2)
        plt.text((x[i] + x[i+1])/2, max(y[i], y[i+1]) + 0.03, '*', 
                 color='green', ha='center', va='bottom', fontsize=15)
    else:
        # Non-significant difference
        plt.plot([x[i] + 0.4, x[i+1] - 0.4], [max(y[i], y[i+1]) + 0.02, max(y[i], y[i+1]) + 0.02], 
                 'r--', linewidth=2)
        plt.text((x[i] + x[i+1])/2, max(y[i], y[i+1]) + 0.03, 'NS', 
                 color='red', ha='center', va='bottom', fontsize=9)

plt.xticks(x, het_test_fisher.figure_data['x'])
plt.title(het_test_fisher.figure_data['title'])
plt.xlabel(het_test_fisher.figure_data['xlabel'])
plt.ylabel(het_test_fisher.figure_data['ylabel'])
plt.ylim(0, max(y) * 1.2)  # Leave room for annotations

# Add annotations
for i, annotation in enumerate(het_test_fisher.figure_data['annotations']):
    plt.annotate(annotation, xy=(0.02, 0.95-i*0.05), xycoords='axes fraction',
                 fontsize=9, ha='left', va='top')

# Plot 2: Default rates by rating with confidence intervals
plt.subplot(2, 1, 2)
# Calculate confidence intervals for each rating's default rate
confidence_intervals = []
for stat in het_test_fisher.details['rating_stats']:
    n = stat['n_obs']
    p = stat['default_rate']
    # Wilson score interval
    if n > 0:
        z = 1.96  # 95% confidence
        denominator = 1 + z**2/n
        center = (p + z**2/(2*n)) / denominator
        interval = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
        lower = max(0, center - interval)
        upper = min(1, center + interval)
    else:
        lower, upper = 0, 0
    confidence_intervals.append((lower, upper))

# Plot with error bars
y = [stat['default_rate'] for stat in het_test_fisher.details['rating_stats']]
ci_errors = np.array([(y[i] - ci[0], ci[1] - y[i]) for i, ci in enumerate(confidence_intervals)]).T

plt.errorbar(x, y, yerr=ci_errors, fmt='o-', capsize=5, elinewidth=2, markersize=8)

# Add target default rates for comparison
target_drs = [target_dr_by_rating[int(r)] for r in het_test_fisher.figure_data['x']]
plt.plot(x, target_drs, 'r--', label='Target Default Rate')

plt.xticks(x, het_test_fisher.figure_data['x'])
plt.title('Default Rates by Rating Grade with 95% Confidence Intervals')
plt.xlabel('Rating Grade')
plt.ylabel('Default Rate')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('rating_heterogeneity_test_example.png', dpi=300)
plt.show()

print("\nExample completed. Results saved to 'rating_heterogeneity_test_example.png'")
