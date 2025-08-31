"""
Integration tests for CRMStudio.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from crmstudio.metrics.pd_metrics import AUC, ROCCurve, Gini
from crmstudio.core.plotting import PlottingService

class TestMetricsPipeline:
    """Integration tests for the metrics pipeline."""
    
    def test_compute_and_plot_workflow(self, binary_test_data, tmp_path):
        """Test the complete workflow from data to metrics to plotting."""
        y_true, y_pred, _ = binary_test_data
        
        # 1. Compute AUC
        auc = AUC("test_model")
        auc_result = auc.compute(y_true=y_true, y_pred=y_pred)
        
        # 2. Compute ROC Curve
        roc = ROCCurve("test_model")
        roc_result = roc.compute(y_true=y_true, y_pred=y_pred)
        
        # 3. Compute Gini
        gini = Gini("test_model")
        gini_result = gini.compute(y_true=y_true, y_pred=y_pred)
        
        # 4. Check results are consistent
        assert auc_result.value == roc_result.value  # AUC and ROC should have same value
        assert abs(gini_result.value - (2 * auc_result.value - 1)) < 1e-10  # Gini = 2*AUC - 1
        
        # 5. Plot ROC curve and save to file
        output_file = os.path.join(tmp_path, "roc_curve.png")
        roc.save_plot(output_file, roc_result)
        
        # 6. Verify file was created
        assert os.path.exists(output_file)
        assert os.path.getsize(output_file) > 0
    
    def test_subpopulation_analysis_workflow(self, time_series_test_data, tmp_path):
        """Test the subpopulation analysis workflow."""
        df = time_series_test_data
        
        # 1. Compute AUC by segment
        auc = AUC("test_model")
        segment_result = auc.compute_by_segment(
            y_true=df['y_true'], 
            y_pred=df['y_pred'], 
            segments=df['segment']
        )
        
        # 2. Compute AUC over time
        time_result = auc.compute_over_time(
            y_true=df['y_true'], 
            y_pred=df['y_pred'], 
            time_index=df['date'], 
            freq='Q'
        )
        
        # 3. Verify results
        assert 'results_df' in segment_result.figure_data
        assert 'results_df' in time_result.figure_data
        
        # 4. Save plots
        segment_file = os.path.join(tmp_path, "auc_by_segment.png")
        time_file = os.path.join(tmp_path, "auc_over_time.png")
        
        service = PlottingService()
        
        # Get and save images
        segment_image = service.plot(segment_result)
        service.save_image(segment_image, segment_file)
        
        time_image = service.plot(time_result)
        service.save_image(time_image, time_file)
        
        # 5. Verify files were created
        assert os.path.exists(segment_file)
        assert os.path.exists(time_file)
        assert os.path.getsize(segment_file) > 0
        assert os.path.getsize(time_file) > 0
