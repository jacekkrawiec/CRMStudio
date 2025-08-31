"""
Shared data classes for CRMStudio.

This module contains data classes used across the CRMStudio package
to ensure consistent data structures and avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class MetricResult:
    """
    Unified result class for all metrics.
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
    
    def __repr__(self) -> str:
        if self.has_figure():
            return f"<MetricResult {self.name} (figure)>"
        else:
            return f"<MetricResult {self.name}: {self.value:.4f}, passes = {self.passed}>"
