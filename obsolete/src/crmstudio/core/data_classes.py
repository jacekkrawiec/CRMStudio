"""
Shared data classes for CRMStudio.

This module contains data classes used across the CRMStudio package
to ensure consistent data structures and avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import pandas as pd


@dataclass
class MetricResult:
    """
    Unified result class for all metrics.
    
    This class provides a standardized format for storing metric results,
    including scalar values, thresholds, and figure data. It also provides
    methods for converting figure data to a standardized format for plotting.
    """
    name: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)
    figure_data: Optional[Dict[str, Any]] = None 

    def has_figure(self) -> bool:
        """Check if this result contains figure data"""
        return self.figure_data is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        result = {
            'metric': self.name,
            'type': 'figure' if self.has_figure() else 'scalar'
        }

        if self.value is not None:
            result['value'] = self.value
        
        if self.threshold is not None:
            result['threshold'] = self.threshold
            result['passed'] = self.passed
        
        if self.details:
            result['details'] = self.details

        if self.figure_data:
            result['figure_data'] = self.figure_data

        return result
    
    def prepare_for_plotting(self) -> List[Dict[str, Any]]:
        """
        Standardize figure_data for plotting.
        
        This method converts the figure_data into a standardized format
        that can be directly used by plotting functions. It handles various
        input formats including DataFrames, single datasets, and collections.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of standardized dataset dictionaries ready for plotting
        """
        if not self.has_figure():
            # If no figure data, return a simple value dataset
            return [{"Full sample": self.value}]
        
        datasets = []
        figure_data = self.figure_data
        
        # Case 1: If results_df is present (for group analysis)
        if 'results_df' in figure_data:
            results_df = figure_data['results_df']
            
            # Verify that the required columns exist
            if 'group' not in results_df.columns:
                raise ValueError("results_df must contain a 'group' column")
            
            if 'figure_data' not in results_df.columns and 'value' not in results_df.columns:
                raise ValueError("results_df must contain either 'figure_data' or 'value' column")
            
            # Extract metadata common to all datasets
            metadata = {}
            for meta_key in ['group_type', 'group_name']:
                if meta_key in results_df.columns and not results_df[meta_key].isna().all():
                    metadata[meta_key] = results_df[meta_key].iloc[0]
            
            # Handle time-based indices by converting them to strings
            has_figure_data = 'figure_data' in results_df.columns
            
            for i, row in results_df.iterrows():
                label = row['group']
                # Convert Period or Timestamp objects to strings
                if isinstance(label, (pd.Period, pd.Timestamp)):
                    label = str(label)
                
                if has_figure_data:
                    data = row['figure_data']
                    if data is None and 'value' in row:
                        data = row['value']
                else:
                    data = row['value']
                
                # Add metadata to dataset
                if isinstance(data, dict):
                    for key, value in metadata.items():
                        if key not in data:
                            data[key] = value
                
                datasets.append({label: data})
        else:
            # Case 2: If no results_df, treat as a single dataset
            datasets.append({"Full sample": figure_data})
        
        return datasets
    
    def is_scalar_collection(self) -> bool:
        """
        Determine if this result contains only scalar values.
        
        Returns
        -------
        bool
            True if all datasets contain only scalar values, False otherwise
        """
        datasets = self.prepare_for_plotting()
        
        for dataset in datasets:
            for key, value in dataset.items():
                if isinstance(value, dict):
                    return False
        
        return True
    
    def is_curve_data(self) -> bool:
        """
        Determine if this result contains curve data.
        
        Returns
        -------
        bool
            True if any dataset contains curve data (x/y coordinates), False otherwise
        """
        if not self.has_figure():
            return False
            
        datasets = self.prepare_for_plotting()
        
        for dataset in datasets:
            for key, value in dataset.items():
                if isinstance(value, dict) and 'x' in value and 'y' in value:
                    return True
        
        return False
    
    def __repr__(self) -> str:
        if self.has_figure():
            return f"<MetricResult {self.name} (figure)>"
        elif self.value is not None:
            return f"<MetricResult {self.name}: {self.value:.4f}, passes = {self.passed}>"
        else:
            return f"<MetricResult {self.name}>"
