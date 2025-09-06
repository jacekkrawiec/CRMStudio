"""
Example demonstrating the use of RatingHomogeneityTest for credit risk models.

This example shows how to use the RatingHomogeneityTest class to assess whether
default rates are homogeneous within each rating grade. The test evaluates if there
are significant differences in default rates within the same rating grade across
different segments (defined by PD buckets).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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
# Well-differentiated base default rates
base_dr_by_rating = {
    1: 0.01,  # 1% default rate for rating 1
    2: 0.03,  # 3% default rate for rating 2
    3: 0.08,  # 8% default rate for rating 3
    4: 0.15,  # 15% default rate for rating 4
    5: 0.30   # 30% default rate for rating 5
}

# Create a synthetic factor that causes heterogeneity within ratings
# This simulates a risk driver not fully captured by the rating model
# For example, this could be industry, region, or some other risk characteristic
heterogeneity_factor = np.random.uniform(0, 1, n_samples)

# Define which ratings will have homogeneity issues
# In this example, ratings 2 and 4 will have heterogeneity
heterogeneous_ratings = [2, 4]

# Generate actual defaults based on rating-specific default rates,
# but with heterogeneity within specific rating grades
y_true = []
y_pred = []

for i in range(n_samples):
    rating = ratings[i]
    base_dr = base_dr_by_rating[rating]
    
    # For heterogeneous ratings, adjust default probability based on the heterogeneity factor
    if rating in heterogeneous_ratings:
        # Create a clear difference within the rating grade
        if heterogeneity_factor[i] < 0.5:
            # Lower half gets lower default rate
            actual_dr = base_dr * 0.5
            # Predicted PD doesn't fully capture this difference
            pred_pd = base_dr * 0.8
        else:
            # Upper half gets higher default rate
            actual_dr = base_dr * 1.5
            # Predicted PD doesn't fully capture this difference
            pred_pd = base_dr * 1.2
    else:
        # Homogeneous ratings have consistent default rates
        actual_dr = base_dr
        # Add small random noise to predicted PDs
        pred_pd = base_dr * (1 + 0.1 * (np.random.random() - 0.5))
    
    # Generate binary default indicator
    default = np.random.binomial(1, actual_dr)
    y_true.append(default)
    y_pred.append(pred_pd)

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Ensure predicted PDs are between 0 and 1
y_pred = np.clip(y_pred, 0.001, 0.999)

# Create a DataFrame for easier analysis
df = pd.DataFrame({
    'rating': ratings,
    'actual': y_true,
    'predicted': y_pred,
    'heterogeneity_factor': heterogeneity_factor
})

# Print basic statistics
print("Dataset Statistics:")
print(f"Total observations: {n_samples}")
print(f"Overall default rate: {y_true.mean():.2%}")
print(f"Average predicted PD: {y_pred.mean():.2%}")

# Print statistics by rating grade
print("\nRating Grade Statistics:")
rating_stats = df.groupby('rating').agg({
    'actual': ['count', 'mean'],
    'predicted': 'mean'
}).reset_index()
rating_stats.columns = ['Rating', 'Count', 'Observed DR', 'Predicted PD']
print(rating_stats.to_string(index=False, float_format=lambda x: f"{x:.2%}" if x < 1 else f"{int(x)}"))

# Print statistics by rating grade and heterogeneity factor bucket (to demonstrate heterogeneity)
print("\nHeterogeneity Analysis - Default Rates by Rating and PD Bucket:")
# Add PD bucket
df['pd_bucket'] = pd.qcut(df['predicted'], 5, labels=False, duplicates='drop')

# Group by rating and PD bucket
het_analysis = df.groupby(['rating', 'pd_bucket']).agg({
    'actual': ['count', 'mean'],
    'predicted': 'mean'
}).reset_index()

het_analysis.columns = ['Rating', 'PD Bucket', 'Count', 'Observed DR', 'Avg PD']
print(het_analysis.to_string(index=False, float_format=lambda x: f"{x:.2%}" if x < 1 else f"{int(x)}"))

# Run rating homogeneity test
hom_test = calibration.RatingHomogeneityTest("Example Model").compute(
    y_true=y_true, 
    y_pred=y_pred, 
    ratings=ratings, 
    n_buckets=5  # Split each rating into 5 buckets by PD
)

# Print homogeneity test results
print("\nRating Homogeneity Test Results:")
print(f"Proportion of homogeneous ratings: {hom_test.value:.2%}")
print(f"Test passed (all ratings homogeneous): {hom_test.passed}")

print("\nRating-level results:")
for result in hom_test.details['rating_results']:
    print(f"Rating {result['rating']}: " +
          f"p-value: {result['p_value']:.4f}, " +
          f"Passed: {result['passed']}, " +
          f"DR Range: {result['min_dr']:.2%} - {result['max_dr']:.2%}, " +
          f"Relative Range: {result['relative_range']:.2f}x")

# Create a custom visualization of the homogeneity test results
plt.figure(figsize=(16, 12))

# Plot 1: p-values by rating grade
plt.subplot(2, 2, 1)
x = [int(float(r)) for r in hom_test.figure_data['x']]
y = hom_test.figure_data['y']
passed = hom_test.figure_data['passed']

# Create color-coded bars based on pass/fail
colors = ['green' if p else 'red' for p in passed]
bars = plt.bar(x, y, color=colors, alpha=0.7)

# Add significance threshold line
alpha = 0.05  # Assuming 95% confidence level
plt.axhline(y=alpha, color='red', linestyle='--', label=f'α = {alpha}')

plt.title('Homogeneity Test p-values by Rating Grade')
plt.xlabel('Rating Grade')
plt.ylabel('p-value')
plt.xticks(x)
plt.legend()

# Annotate bars with pass/fail
for i, (bar, p) in enumerate(zip(bars, passed)):
    plt.text(bar.get_x() + bar.get_width()/2., 0.01, 
             'PASS' if p else 'FAIL', 
             color='white', ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle="round", fc=colors[i], ec="white", alpha=0.8))

# Plot 2: Default rate ranges within each rating
plt.subplot(2, 2, 2)
rating_results = hom_test.details['rating_results']

# Prepare data
ratings = [int(float(r['rating'])) for r in rating_results]
min_drs = [r['min_dr'] for r in rating_results]
max_drs = [r['max_dr'] for r in rating_results]
overall_drs = [r['overall_dr'] for r in rating_results]

# Plot default rate ranges
for i, rating in enumerate(ratings):
    plt.plot([rating, rating], [min_drs[i], max_drs[i]], 'o-', linewidth=2,
             color='red' if not rating_results[i]['passed'] else 'green',
             markersize=8)
    # Add overall DR marker
    plt.plot(rating, overall_drs[i], 'Xk', markersize=10)

plt.title('Default Rate Ranges within Rating Grades')
plt.xlabel('Rating Grade')
plt.ylabel('Default Rate')
plt.xticks(ratings)
plt.grid(True, linestyle='--', alpha=0.5)

# Create a custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='green', label='Homogeneous Rating',
           markersize=8, linestyle='-', linewidth=2),
    Line2D([0], [0], marker='o', color='red', label='Heterogeneous Rating',
           markersize=8, linestyle='-', linewidth=2),
    Line2D([0], [0], marker='X', color='black', label='Overall Default Rate',
           markersize=10, linestyle='')
]
plt.legend(handles=legend_elements)

# Plot 3: Detailed view of bucket-level default rates for heterogeneous ratings
plt.subplot(2, 1, 2)

# Extract buckets data for visualization
buckets_data = pd.DataFrame(hom_test.details['buckets_data'])

# Create a pivot table with ratings as columns and buckets as rows
pivot_data = buckets_data.pivot_table(
    index='bucket', 
    columns='rating',
    values='default_rate',
    aggfunc='mean'
)

# Plot heatmap of default rates by rating and bucket
plt.imshow(pivot_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
plt.colorbar(label='Default Rate')
plt.title('Default Rates by Rating Grade and PD Bucket')
plt.xlabel('Rating Grade')
plt.ylabel('PD Bucket (0 = Lowest PD, 4 = Highest PD)')

# Set x and y tick labels
plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
plt.yticks(range(len(pivot_data.index)), pivot_data.index)

# Add text annotations with default rates
for i in range(len(pivot_data.index)):
    for j in range(len(pivot_data.columns)):
        if not np.isnan(pivot_data.iloc[i, j]):
            plt.text(j, i, f"{pivot_data.iloc[i, j]:.2%}", 
                     ha="center", va="center", color="black" if pivot_data.iloc[i, j] < 0.15 else "white")

plt.tight_layout()
plt.savefig('rating_homogeneity_test_example.png', dpi=300)
plt.show()

print("\nExample completed. Results saved to 'rating_homogeneity_test_example.png'")

# Demonstrate how to use the PlottingService to visualize the results
from crmstudio.core.plotting import PlottingService

# Create plotting service instance
plotter = PlottingService()

# Generate plot using the MetricResult directly
homogeneity_plot = plotter.plot(hom_test, plot_type='calibration')

# Save the generated plot
plotter.save_image(homogeneity_plot, 'rating_homogeneity_service_plot.png')

print("Plotting service visualization saved to 'rating_homogeneity_service_plot.png'")
