"""
Example demonstrating the use of concentration metrics in CRMStudio.

This example shows how to:
1. Calculate the Herfindahl Index for rating grade concentration
2. Analyze the distribution of exposures across rating grades
3. Interpret concentration levels using standard thresholds

Concentration analysis is essential for regulatory compliance and risk management,
as excessive concentration in a few rating grades may indicate issues with the
rating model or process.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Import CRMStudio metrics
from crmstudio.metrics.pd import calibration
import importlib

importlib.reload(calibration)

# Set random seed for reproducibility
np.random.seed(42)

# Create plots directory if it doesn't exist
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Example 1: Highly Concentrated Portfolio
print("\nExample 1: Highly Concentrated Portfolio")
print("-" * 50)

# Generate synthetic rating data (1 = best, 10 = worst)
n_obligors = 1000
rating_scale = list(range(1, 11))

# Create a concentrated distribution with most obligors in a few rating grades
concentrated_ratings = np.random.choice(
    rating_scale, 
    size=n_obligors,
    p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.20, 0.30, 0.10]  # Higher concentration in grades 8-9
)

# Generate exposure values (higher exposures for better ratings)
exposures = np.zeros(n_obligors)
for i, rating in enumerate(concentrated_ratings):
    # Base exposure with some randomness
    base = 1000000 * (1 / rating)  # Better ratings get higher exposures
    variation = np.random.uniform(0.5, 1.5)
    exposures[i] = base * variation

# Calculate Herfindahl Index
hi_concentrated = calibration.HerfindahlIndex("example_model")
hi_concentrated_result = hi_concentrated.compute(
    ratings=concentrated_ratings,
    exposures=exposures
)

# Print results
print(f"Number of rating grades: {hi_concentrated_result.details['n_ratings']}")
print(f"Herfindahl Index: {hi_concentrated_result.value:.4f}")
print(f"Normalized HI: {hi_concentrated_result.details['hi_normalized']:.4f}")
print(f"Concentration category: {hi_concentrated_result.details['concentration_category']}")
print(f"Effective number of grades: {hi_concentrated_result.details['effective_n']:.2f} out of {hi_concentrated_result.details['n_ratings']}")
print(f"Top 3 concentration: {hi_concentrated_result.details['top_3_concentration']:.2%}")
print(f"Top 5 concentration: {hi_concentrated_result.details['top_5_concentration']:.2%}")

print("\nTop 5 rating grades by exposure:")
for i, stat in enumerate(hi_concentrated_result.details['rating_stats'][:5]):
    print(f"  Rating {stat['rating']}: {stat['proportion']:.2%} of total exposure")

# Show and save the plot
hi_concentrated.show_plot()
hi_concentrated.save_plot(os.path.join(plots_dir, 'concentrated_portfolio.png'))

# Example 2: Well-Distributed Portfolio
print("\n\nExample 2: Well-Distributed Portfolio")
print("-" * 50)

# Create a more evenly distributed portfolio
well_distributed_ratings = np.random.choice(
    rating_scale, 
    size=n_obligors,
    p=[0.08, 0.09, 0.10, 0.12, 0.12, 0.12, 0.12, 0.10, 0.09, 0.06]  # More even distribution
)

# Generate exposure values
even_exposures = np.zeros(n_obligors)
for i, rating in enumerate(well_distributed_ratings):
    # Base exposure with some randomness
    base = 1000000 * (1 / rating)  # Better ratings get higher exposures
    variation = np.random.uniform(0.8, 1.2)  # Less variation
    even_exposures[i] = base * variation

# Calculate Herfindahl Index
hi_distributed = calibration.HerfindahlIndex("example_model")
hi_distributed_result = hi_distributed.compute(
    ratings=well_distributed_ratings,
    exposures=even_exposures
)

# Print results
print(f"Number of rating grades: {hi_distributed_result.details['n_ratings']}")
print(f"Herfindahl Index: {hi_distributed_result.value:.4f}")
print(f"Normalized HI: {hi_distributed_result.details['hi_normalized']:.4f}")
print(f"Concentration category: {hi_distributed_result.details['concentration_category']}")
print(f"Effective number of grades: {hi_distributed_result.details['effective_n']:.2f} out of {hi_distributed_result.details['n_ratings']}")
print(f"Top 3 concentration: {hi_distributed_result.details['top_3_concentration']:.2%}")
print(f"Top 5 concentration: {hi_distributed_result.details['top_5_concentration']:.2%}")

print("\nTop 5 rating grades by exposure:")
for i, stat in enumerate(hi_distributed_result.details['rating_stats'][:5]):
    print(f"  Rating {stat['rating']}: {stat['proportion']:.2%} of total exposure")

# Show and save the plot
hi_distributed.show_plot()
hi_distributed.save_plot(os.path.join(plots_dir, 'distributed_portfolio.png'))

# Example 3: Comparing by Number of Observations vs. Exposure
print("\n\nExample 3: Comparing by Number of Observations vs. Exposure")
print("-" * 50)

# Use the well-distributed ratings from Example 2
# But now we'll compare HI by count vs. by exposure

# Calculate HI using equal weights (count-based)
hi_count = calibration.HerfindahlIndex("example_model")
hi_count_result = hi_count.compute(
    ratings=well_distributed_ratings
    # No exposures provided, so each observation has equal weight
)

# Calculate HI using exposures (already done in Example 2)
print("Concentration by number of obligors:")
print(f"Herfindahl Index: {hi_count_result.value:.4f}")
print(f"Normalized HI: {hi_count_result.details['hi_normalized']:.4f}")
print(f"Concentration category: {hi_count_result.details['concentration_category']}")

print("\nConcentration by exposure amount:")
print(f"Herfindahl Index: {hi_distributed_result.value:.4f}")
print(f"Normalized HI: {hi_distributed_result.details['hi_normalized']:.4f}")
print(f"Concentration category: {hi_distributed_result.details['concentration_category']}")

print("\nComparison demonstrates that concentration can appear different")
print("when measured by number of obligors versus by exposure amount.")
print("Regulatory focus is typically on exposure concentration.")

# Example 4: Effect of changing threshold
print("\n\nExample 4: Effect of changing threshold")
print("-" * 50)

# Set a custom threshold to demonstrate passing/failing
custom_threshold = 0.25
hi_custom = calibration.HerfindahlIndex("example_model", config={"hi_threshold": custom_threshold})
hi_custom_result = hi_custom.compute(
    ratings=concentrated_ratings,
    exposures=exposures
)

print(f"Custom threshold: {custom_threshold}")
print(f"Herfindahl Index: {hi_custom_result.value:.4f}")
print(f"Test passed: {hi_custom_result.passed}")

print("\nExample completed successfully.")
