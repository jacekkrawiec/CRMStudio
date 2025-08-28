"""
Test the refactored plotting functionality
"""

import numpy as np
import sys
import os
import importlib
from pathlib import Path

# Add the src directory to the path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules after path setup
from crmstudio.metrics.pd_metrics import ROCCurve, CAPCurve, KSDistPlot
from crmstudio.core.base import BaseMetric
import pandas as pd

def generate_test_data(n_samples=10000):
    """Generate synthetic test data for metric calculation"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate target variable
    y_true = np.random.binomial(1, 0.1, size=n_samples)  # 10% positives
    
    # Generate predicted scores (higher for positives)
    y_pred = np.where(y_true == 1,
                    np.random.beta(5, 2, size=n_samples),  # Scores for positive cases
                    np.random.beta(2, 5, size=n_samples))  # Scores for negative cases
    
    # Generate a segment variable
    segments = np.random.choice(['A', 'B', 'C'], size=n_samples)
    
    # Generate a time variable
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='M')
    time_var = np.random.choice(dates, size=n_samples)
    
    return y_true, y_pred, segments, time_var

def test_curve_plotting():
    """Test curve plotting functionality"""
    print("Testing curve plotting...")
    
    # Generate test data
    y_true, y_pred, segments, time_var = generate_test_data()
    
    # Test ROC Curve
    roc = ROCCurve("test_model")
    result = roc.compute(y_true=y_true, y_pred=y_pred)
    print(f"ROC AUC: {result.value:.4f}")
    
    # Test plotting functionality
    print("Plotting ROC curve...")
    roc.show_plot(result.figure_data)
    
    # Test segment analysis
    print("Testing segment analysis...")
    segment_results = roc.compute_by_segment(y_true=y_true, y_pred=y_pred, segments=segments)
    print(segment_results[['group', 'value']])
    
    # Test group plotting
    print("Plotting segment analysis...")
    roc.show_group_plot(segment_results)
    
    # Test time analysis
    print("Testing time analysis...")
    time_results = roc.compute_over_time(y_true=y_true, y_pred=y_pred, time_index=time_var, freq='Q')
    print(time_results[['group', 'value']])
    
    # Test group plotting for time
    print("Plotting time analysis...")
    roc.show_group_plot(time_results)
    
    print("All curve plotting tests passed!")

def test_distribution_plotting():
    """Test distribution plotting functionality"""
    print("Testing distribution plotting...")
    
    # Generate test data
    y_true, y_pred, segments, time_var = generate_test_data()
    
    # Test KS Distribution Plot
    ks = KSDistPlot("test_model")
    result = ks.compute(y_true=y_true, y_pred=y_pred)
    print(f"KS value: {result.value:.4f}")
    
    # Test plotting functionality
    print("Plotting KS distribution...")
    ks.show_plot(result.figure_data)
    
    # Test segment analysis
    print("Testing segment analysis...")
    segment_results = ks.compute_by_segment(y_true=y_true, y_pred=y_pred, segments=segments)
    print(segment_results[['group', 'value']])
    
    # Test group plotting
    print("Plotting segment analysis...")
    ks.show_group_plot(segment_results)
    
    print("All distribution plotting tests passed!")

if __name__ == "__main__":
    print("Running refactoring tests...")
    test_curve_plotting()
    test_distribution_plotting()
    print("All tests completed successfully!")
