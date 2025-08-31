"""
Unit tests for core components of CRMStudio.
"""

import pytest
import numpy as np
import pandas as pd
from crmstudio.core.data_classes import MetricResult
from crmstudio.core.plotting import PlottingService
from crmstudio.metrics.pd_metrics import AUC, ROCCurve

class TestMetricResult:
    """Tests for MetricResult data class."""
    
    def test_metric_result_creation(self):
        """Test basic creation of MetricResult."""
        result = MetricResult(name="TestMetric", value=0.8, threshold=0.7, passed=True)
        
        assert result.name == "TestMetric"
        assert result.value == 0.8
        assert result.threshold == 0.7
        assert result.passed is True
        assert result.details == {}
        assert result.figure_data is None
    
    def test_has_figure(self):
        """Test has_figure method."""
        result1 = MetricResult(name="TestMetric", value=0.8)
        assert result1.has_figure() is False
        
        result2 = MetricResult(name="TestMetric", figure_data={"x": [0, 1], "y": [0, 1]})
        assert result2.has_figure() is True
    
    def test_to_dict(self):
        """Test to_dict method."""
        result = MetricResult(
            name="TestMetric", 
            value=0.8, 
            threshold=0.7, 
            passed=True,
            details={"n_obs": 100},
            figure_data={"x": [0, 1], "y": [0, 1]}
        )
        
        data_dict = result.to_dict()
        
        assert data_dict["metric"] == "TestMetric"
        assert data_dict["type"] == "figure"
        assert data_dict["value"] == 0.8
        assert data_dict["threshold"] == 0.7
        assert data_dict["passed"] is True
        assert data_dict["details"]["n_obs"] == 100
        assert "figure_data" in data_dict

class TestPlottingService:
    """Tests for PlottingService."""
    
    def test_plotting_service_initialization(self):
        """Test basic initialization of PlottingService."""
        service = PlottingService()
        assert service.style is not None
    
    def test_set_style(self):
        """Test setting custom style."""
        service = PlottingService()
        custom_style = {"colors": {"primary": "#FF0000"}}
        
        service.set_style(custom_style)
        assert service.style == custom_style
    
    def test_plot_with_curve_metric(self, binary_test_data):
        """Test plotting a curve metric."""
        y_true, y_pred, _ = binary_test_data
        
        roc = ROCCurve("test_model")
        result = roc.compute(y_true=y_true, y_pred=y_pred)
        
        service = PlottingService()
        image_data = service.plot(result, plot_type="curve")
        
        assert "image_base64" in image_data
        assert "width" in image_data
        assert "height" in image_data
    
    def test_plot_with_scalar_metric(self, binary_test_data):
        """Test plotting a scalar metric."""
        y_true, y_pred, _ = binary_test_data
        
        auc = AUC("test_model")
        result = auc.compute(y_true=y_true, y_pred=y_pred)
        
        service = PlottingService()
        image_data = service.plot(result)
        
        assert "image_base64" in image_data
