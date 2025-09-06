"""
Configuration file for pytest with shared fixtures for CRMStudio tests.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

@pytest.fixture(scope="session")
def binary_test_data():
    """
    Fixture that provides binary classification test data with known characteristics.
    
    Returns:
        tuple: (y_true, y_pred, expected_auc)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample data with known characteristics
    n_samples = 1000
    
    # Ground truth with 20% positive rate
    y_true = np.zeros(n_samples)
    y_true[:200] = 1
    
    # Create predictions with known AUC (by designing the distributions)
    # Perfect separation would have all positives with higher scores than negatives
    # We'll create predictions where positives generally have higher scores but with some overlap
    
    # Positives from Beta(8, 2) - skewed high
    pos_scores = np.random.beta(8, 2, size=200)
    # Negatives from Beta(2, 8) - skewed low
    neg_scores = np.random.beta(2, 8, size=800)
    
    # Combine and shuffle
    y_pred = np.zeros(n_samples)
    y_pred[:200] = pos_scores
    y_pred[200:] = neg_scores
    
    # Calculate expected AUC (approximate) - not exact due to random generation
    # but should be close to 0.95 with this configuration
    expected_auc = 0.95
    
    return y_true, y_pred, expected_auc

@pytest.fixture(scope="session")
def time_series_test_data():
    """
    Fixture that provides test data with time series information.
    
    Returns:
        pandas.DataFrame: DataFrame with time series binary classification data
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create time periods across 3 years (quarterly)
    dates = []
    for year in range(2020, 2023):
        for quarter in range(1, 5):
            # Create dates at the end of each quarter
            month = quarter * 3
            dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))
    
    # Repeat dates to fill n_samples
    dates_array = np.resize(dates, n_samples)
    
    # Create different default rates per time period to simulate trends
    default_rates = {date: rate for date, rate in zip(dates, np.linspace(0.05, 0.25, len(dates)))}
    
    # Generate y_true based on default rates
    y_true = np.zeros(n_samples)
    for i, date in enumerate(dates_array):
        if np.random.random() < default_rates[date]:
            y_true[i] = 1
    
    # Generate y_pred with varying quality over time
    # Early periods have better prediction quality
    y_pred = np.zeros(n_samples)
    for i, date in enumerate(dates_array):
        idx = dates.index(date)
        separation_quality = 1 - (idx / len(dates)) * 0.5  # Ranges from 1.0 to 0.5
        
        if y_true[i] == 1:
            # Positive class (defaulted)
            y_pred[i] = np.random.beta(8 * separation_quality, 2, size=1)[0]
        else:
            # Negative class (non-defaulted)
            y_pred[i] = np.random.beta(2, 8 * separation_quality, size=1)[0]
    
    # Create segment information (4 segments)
    segments = np.random.choice(['Segment_A', 'Segment_B', 'Segment_C', 'Segment_D'], size=n_samples)
    
    # Create a continuous variable for range-based analysis
    continuous_var = np.random.normal(50000, 15000, size=n_samples)
    
    # Combine into DataFrame
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'date': dates_array,
        'segment': segments,
        'continuous_var': continuous_var
    })
    
    return df

@pytest.fixture(scope="session")
def config_fixture():
    """
    Fixture that provides a basic configuration for testing.
    
    Returns:
        dict: Configuration dictionary
    """
    return {
        "models": {
            "test_model": {
                "metrics": [
                    {
                        "name": "auc",
                        "threshold": 0.7,
                        "params": {}
                    },
                    {
                        "name": "roc_curve",
                        "produce_figure": True,
                        "params": {}
                    },
                    {
                        "name": "gini",
                        "threshold": 0.4,
                        "params": {}
                    }
                ]
            }
        },
        "report": {
            "output_dir": "test_reports/"
        }
    }
