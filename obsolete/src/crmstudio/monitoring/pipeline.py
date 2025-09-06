"""
Monitoring pipeline for running and evaluating multiple metrics.

This module provides the main pipeline for model monitoring, which:
1. Loads configuration from a YAML file
2. Processes input data according to specifications
3. Runs appropriate metrics based on configuration
4. Generates reports and triggers alerts if thresholds are exceeded
"""

import os
import yaml
import pandas as pd
import numpy as np
import json
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import importlib
import matplotlib.pyplot as plt

# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)

from ..core.config_loader import load_config

# Import discrimination metrics
from ..metrics.pd.discrimination import (
    AUC, AUCDelta, ROCCurve, PietraIndex, KSStat, Gini, GiniCI, 
    CAPCurve, CIER, KLDistance, InformationValue, KendallsTau, 
    SpearmansRho, KSDistPlot, ScoreHistogram, PDLiftPlot, PDGainPlot
)

# Import calibration metrics
from ..metrics.pd.calibration import (
    HosmerLemeshow, CalibrationCurve, BrierScore, ExpectedCalibrationError,
    BinomialTest, NormalTest, JeffreysTest, RatingHeterogeneityTest,
    HerfindahlIndex
)

# Import stability metrics
from ..metrics.pd.stability import (
    PSI, CSI, TemporalDriftDetection, MigrationAnalysis, RatingStabilityAnalysis
)
from ..core.data_classes import MetricResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringPipeline:
    """
    Main pipeline for running model monitoring.
    
    This class handles the entire monitoring process:
    1. Loading configuration
    2. Processing data
    3. Running metrics
    4. Generating reports
    5. Triggering alerts
    
    Attributes
    ----------
    config : dict
        Configuration dictionary loaded from YAML
    results_dir : str
        Directory for saving results
    report_dir : str
        Directory for saving reports
    """
    
    # Mapping of metric names to their class implementations
    METRIC_MAPPING = {
        # Discrimination metrics
        'auc': AUC,
        'auc_delta': AUCDelta,
        'roc_curve': ROCCurve, 
        'pietra_index': PietraIndex,
        'ks_stat': KSStat,
        'gini': Gini,
        'gini_ci': GiniCI,
        'cap_curve': CAPCurve,
        'cier': CIER,
        'kl_distance': KLDistance,
        'information_value': InformationValue,
        'kendalls_tau': KendallsTau,
        'spearmans_rho': SpearmansRho,
        'ks_dist_plot': KSDistPlot,
        'score_histogram': ScoreHistogram,
        'pd_lift_plot': PDLiftPlot,
        'pd_gain_plot': PDGainPlot,
        
        # Calibration metrics
        'hosmer_lemeshow': HosmerLemeshow,
        'calibration_curve': CalibrationCurve,
        'brier_score': BrierScore,
        'ece': ExpectedCalibrationError,
        'binomial_test': BinomialTest,
        'normal_test': NormalTest,
        'jeffreys_test': JeffreysTest,
        'rating_heterogeneity': RatingHeterogeneityTest,
        'herfindahl_index': HerfindahlIndex,
        
        # Stability metrics
        'psi': PSI,
        'csi': CSI,
        'temporal_drift': TemporalDriftDetection,
        'migration_analysis': MigrationAnalysis,
        'rating_stability': RatingStabilityAnalysis
    }
    
    def __init__(self, config_path: str = None, config: Dict = None, 
                 results_dir: str = 'results', report_dir: str = 'reports'):
        """
        Initialize the monitoring pipeline.
        
        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file
        config : Dict, optional
            Configuration dictionary (alternative to config_path)
        results_dir : str, optional
            Directory for saving results
        report_dir : str, optional
            Directory for saving reports
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)
            
        # Set up directories
        self.results_dir = results_dir
        self.report_dir = report_dir
        
        # Create directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {}
        self.alerts = []
        
        # Set timestamp for this run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized monitoring pipeline with config: {config_path or 'provided dict'}")
    
    def run(self, data: Optional[Dict[str, pd.DataFrame]] = None, 
            save_results: bool = True, 
            generate_report: bool = True) -> Dict[str, Any]:
        """
        Run the monitoring pipeline.
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame], optional
            Dictionary of DataFrames for different data types (e.g., {'reference': df1, 'current': df2})
        save_results : bool, optional
            Whether to save results to disk
        generate_report : bool, optional
            Whether to generate reports
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of results for all metrics
        """
        logger.info("Starting monitoring pipeline run")
        
        # Process data if provided, otherwise assume it will be loaded in each metric
        self.data = data
        
        # Iterate through models in config
        for model_name, model_config in self.config.get('models', {}).items():
            logger.info(f"Processing model: {model_name}")
            
            # Run metrics for this model
            model_results = self._run_model_metrics(model_name, model_config)
            
            # Ensure all metric results have properly serializable values
            self._ensure_serializable_results(model_results)
            
            # Store results
            self.results[model_name] = model_results
            
            # Check thresholds and trigger alerts
            self._check_thresholds(model_name, model_results, model_config)
            
        # Save results if requested
        if save_results:
            self._save_results()
            
        # Generate reports if requested
        if generate_report:
            self._generate_reports()
            
        logger.info("Monitoring pipeline run completed")
        
        return self.results
        
    def _ensure_serializable_results(self, model_results):
        """
        Ensure all results have properly serializable values.
        
        Parameters
        ----------
        model_results : Dict
            Dictionary of results for a model
        """
        for metric_name, result_dict in model_results.items():
            if 'error' in result_dict:
                continue
                
            result = result_dict.get('result')
            if result is None:
                continue
                
            # Convert NumPy boolean types to Python boolean
            if result.passed is not None:
                result.passed = bool(result.passed)
                
            # Convert NumPy numeric types to Python numeric types
            if result.value is not None:
                result.value = float(result.value)
                
            if result.threshold is not None:
                result.threshold = float(result.threshold)
    
    def _run_model_metrics(self, model_name: str, model_config: Dict) -> Dict[str, Any]:
        """
        Run all metrics for a specific model.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        model_config : Dict
            Configuration for the model
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of results for this model
        """
        metrics = model_config.get('metrics', [])
        model_results = {}
        
        for metric_config in metrics:
            metric_name = metric_config.get('name')
            metric_params = metric_config.get('params', {})
            
            # Skip if metric name is missing
            if not metric_name:
                logger.warning(f"Skipping metric with missing name in model {model_name}")
                continue
                
            # Get metric class from mapping
            metric_class = self.METRIC_MAPPING.get(metric_name.lower())
            if metric_class is None:
                logger.warning(f"Unknown metric: {metric_name} for model {model_name}")
                continue
            
            logger.info(f"Running metric: {metric_name} for model {model_name}")
            
            try:
                # Initialize metric
                metric_instance = metric_class(model_name=model_name, config=metric_params)
                
                # Determine metric type to handle data correctly
                metric_type = self._get_metric_type(metric_name)
                
                # Get the appropriate compute method based on metric type
                compute_method = getattr(
                    self, f"_compute_{metric_type}", self._compute_generic
                )
                
                # Run the compute method with the right parameters
                result = compute_method(metric_instance, model_name, metric_name, metric_params)
                
                # Store the result
                model_results[metric_name] = {
                    'result': result,
                    'config': metric_config,
                    'timestamp': self.timestamp
                }
                
                logger.info(f"Metric {metric_name} completed with value: {result.value}")
                
            except Exception as e:
                logger.error(f"Error running metric {metric_name}: {str(e)}", exc_info=True)
                model_results[metric_name] = {
                    'error': str(e),
                    'config': metric_config,
                    'timestamp': self.timestamp
                }
        
        return model_results
    
    def _get_metric_type(self, metric_name: str) -> str:
        """
        Determine the type of metric based on its name.
        
        Parameters
        ----------
        metric_name : str
            Name of the metric
            
        Returns
        -------
        str
            Type of metric ('discrimination', 'calibration', 'stability')
        """
        metric_lower = metric_name.lower()
        
        # Check discrimination metrics
        discrimination_keywords = [
            'auc', 'roc', 'pietra', 'ks', 'gini', 'cap', 'cier', 
            'kl_distance', 'information_value', 'kendall', 'spearman',
            'histogram', 'lift', 'gain'
        ]
        
        # Check calibration metrics
        calibration_keywords = [
            'hosmer', 'calibration', 'brier', 'ece', 'binomial', 'normal',
            'jeffreys', 'heterogeneity', 'herfindahl'
        ]
        
        # Check stability metrics
        stability_keywords = [
            'psi', 'csi', 'drift', 'migration', 'stability'
        ]
        
        # Determine metric type
        for keyword in discrimination_keywords:
            if keyword in metric_lower:
                return 'discrimination'
                
        for keyword in calibration_keywords:
            if keyword in metric_lower:
                return 'calibration'
                
        for keyword in stability_keywords:
            if keyword in metric_lower:
                return 'stability'
                
        # Default to generic if no match
        return 'generic'
    
    def _compute_discrimination(self, metric, model_name, metric_name, params) -> MetricResult:
        """
        Compute discrimination metrics.
        
        Parameters
        ----------
        metric : BaseMetric
            Metric instance
        model_name : str
            Name of the model
        metric_name : str
            Name of the metric
        params : Dict
            Additional parameters
            
        Returns
        -------
        MetricResult
            Result of the metric computation
        """
        # Check if we have data
        if self.data is None:
            raise ValueError(f"No data provided for discrimination metric {metric_name}")
        
        # Get required data
        y_true = self._get_data_column(model_name, 'y_true', 'target')
        y_pred = self._get_data_column(model_name, 'y_pred', 'score')
        
        # Run the metric
        return metric.compute(y_true=y_true, y_pred=y_pred, **params)
    
    def _compute_calibration(self, metric, model_name, metric_name, params) -> MetricResult:
        """
        Compute calibration metrics.
        
        Parameters
        ----------
        metric : BaseMetric
            Metric instance
        model_name : str
            Name of the model
        metric_name : str
            Name of the metric
        params : Dict
            Additional parameters
            
        Returns
        -------
        MetricResult
            Result of the metric computation
        """
        # Check if we have data
        if self.data is None:
            raise ValueError(f"No data provided for calibration metric {metric_name}")
        
        # Get required data
        y_true = self._get_data_column(model_name, 'y_true', 'target')
        y_pred = self._get_data_column(model_name, 'y_pred', 'score')
        
        # For rating-based metrics, get ratings if available
        ratings = None
        try:
            ratings = self._get_data_column(model_name, 'ratings', 'rating', required=False)
        except:
            pass
        
        # Special case for HerfindahlIndex
        if metric_name.lower() == 'herfindahl_index':
            # Get exposures if available
            exposures = None
            try:
                exposures = self._get_data_column(model_name, 'exposures', 'exposure', required=False)
            except:
                pass
            
            # If we have ratings, compute HerfindahlIndex
            if ratings is not None:
                return metric.compute(ratings=ratings, exposures=exposures, **params)
            else:
                raise ValueError("Ratings data required for HerfindahlIndex")
        
        # Run the metric with ratings if available
        if ratings is not None:
            return metric.compute(y_true=y_true, y_pred=y_pred, ratings=ratings, **params)
        else:
            return metric.compute(y_true=y_true, y_pred=y_pred, **params)
    
    def _compute_stability(self, metric, model_name, metric_name, params) -> MetricResult:
        """
        Compute stability metrics.
        
        Parameters
        ----------
        metric : BaseMetric
            Metric instance
        model_name : str
            Name of the model
        metric_name : str
            Name of the metric
        params : Dict
            Additional parameters
            
        Returns
        -------
        MetricResult
            Result of the metric computation
        """
        # Check if we have data
        if self.data is None:
            raise ValueError(f"No data provided for stability metric {metric_name}")
        
        # Handle different stability metrics
        metric_lower = metric_name.lower()
        
        if metric_lower == 'psi':
            # Get reference and recent scores
            reference_scores = self._get_data_column(model_name, 'reference_scores', 'reference_score')
            recent_scores = self._get_data_column(model_name, 'recent_scores', 'recent_score')
            
            # Run PSI
            return metric.compute(
                reference_scores=reference_scores, 
                recent_scores=recent_scores, 
                **params
            )
            
        elif metric_lower == 'csi':
            # Get reference and recent data
            reference_data = self._get_data(model_name, 'reference_data', 'reference')
            recent_data = self._get_data(model_name, 'recent_data', 'recent')
            
            # Run CSI
            return metric.compute(
                reference_data=reference_data,
                recent_data=recent_data,
                **params
            )
            
        elif metric_lower == 'temporal_drift':
            # Get time points and metric values
            time_points = self._get_data_column(model_name, 'time_points', 'date')
            metric_values = self._get_data_column(model_name, 'metric_values', 'auc')
            
            # Run TemporalDriftDetection
            return metric.compute(
                time_points=time_points,
                metric_values=metric_values,
                metric_name=params.get('metric_name', 'AUC'),
                **params
            )
            
        elif metric_lower == 'migration_analysis':
            # Get initial and current ratings
            initial_ratings = self._get_data_column(model_name, 'initial_ratings', 'initial_rating')
            current_ratings = self._get_data_column(model_name, 'current_ratings', 'current_rating')
            
            # Get rating scale if available
            rating_scale = params.get('rating_scale')
            if rating_scale is None:
                # Try to infer from data
                rating_scale = sorted(list(set(np.concatenate([
                    np.unique(initial_ratings), 
                    np.unique(current_ratings)
                ]))))
            
            # Run MigrationAnalysis
            return metric.compute(
                initial_ratings=initial_ratings,
                current_ratings=current_ratings,
                rating_scale=rating_scale,
                **params
            )
            
        elif metric_lower == 'rating_stability':
            # Get rating time series and time points
            rating_time_series = self._get_data(model_name, 'rating_time_series', 'ratings_over_time')
            
            # If rating_time_series is a DataFrame, convert to list of arrays
            if isinstance(rating_time_series, pd.DataFrame):
                # Assume each column is a time point
                rating_time_series = [rating_time_series[col].values for col in rating_time_series.columns]
            
            # Get time points if available
            time_points = params.get('time_points')
            if time_points is None:
                # Generate time points labels
                time_points = [f"Period {i+1}" for i in range(len(rating_time_series))]
            
            # Run RatingStabilityAnalysis
            return metric.compute(
                rating_time_series=rating_time_series,
                time_points=time_points,
                **params
            )
            
        else:
            raise ValueError(f"Unknown stability metric: {metric_name}")
    
    def _compute_generic(self, metric, model_name, metric_name, params) -> MetricResult:
        """
        Compute generic metrics that don't fit into specific categories.
        
        Parameters
        ----------
        metric : BaseMetric
            Metric instance
        model_name : str
            Name of the model
        metric_name : str
            Name of the metric
        params : Dict
            Additional parameters
            
        Returns
        -------
        MetricResult
            Result of the metric computation
        """
        # Check if we have data
        if self.data is None:
            raise ValueError(f"No data provided for generic metric {metric_name}")
        
        # Get all data for this model
        model_data = self._get_data(model_name)
        
        # Run the metric with all available data
        return metric.compute(**model_data, **params)
    
    def _get_data_column(self, model_name: str, column_name: str, 
                        alt_column_name: str = None, required: bool = True) -> np.ndarray:
        """
        Get a data column from the provided data.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        column_name : str
            Name of the column to retrieve
        alt_column_name : str, optional
            Alternative column name to try
        required : bool
            Whether the column is required
            
        Returns
        -------
        np.ndarray
            Data column
        """
        # Check if we have data
        if self.data is None:
            if required:
                raise ValueError(f"No data provided for model {model_name}")
            else:
                return None
        
        # Try to get model-specific data
        model_data = self.data.get(model_name)
        
        # If no model-specific data, try to use general data
        if model_data is None:
            model_data = self.data
        
        # Try the primary column name
        if isinstance(model_data, dict) and column_name in model_data:
            return np.array(model_data[column_name])
        elif isinstance(model_data, pd.DataFrame) and column_name in model_data.columns:
            return model_data[column_name].values
            
        # Try the alternative column name
        if alt_column_name is not None:
            if isinstance(model_data, dict) and alt_column_name in model_data:
                return np.array(model_data[alt_column_name])
            elif isinstance(model_data, pd.DataFrame) and alt_column_name in model_data.columns:
                return model_data[alt_column_name].values
        
        # If we get here and column is required, raise error
        if required:
            raise ValueError(f"Column {column_name} or {alt_column_name} not found in data for model {model_name}")
        
        return None
    
    def _get_data(self, model_name: str, data_name: str = None, alt_data_name: str = None) -> Any:
        """
        Get data for a specific model.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        data_name : str, optional
            Name of the data to retrieve
        alt_data_name : str, optional
            Alternative data name to try
            
        Returns
        -------
        Any
            Requested data
        """
        # Check if we have data
        if self.data is None:
            raise ValueError(f"No data provided for model {model_name}")
        
        # If no specific data name, return all data for the model
        if data_name is None:
            # Try to get model-specific data
            model_data = self.data.get(model_name)
            
            # If no model-specific data, try to use general data
            if model_data is None:
                model_data = self.data
                
            return model_data
        
        # Try to get model-specific data
        model_data = self.data.get(model_name)
        
        # If no model-specific data, try to use general data
        if model_data is None:
            model_data = self.data
        
        # Try the primary data name
        if isinstance(model_data, dict) and data_name in model_data:
            return model_data[data_name]
            
        # Try the alternative data name
        if alt_data_name is not None:
            if isinstance(model_data, dict) and alt_data_name in model_data:
                return model_data[alt_data_name]
        
        # If we get here, raise error
        raise ValueError(f"Data {data_name} or {alt_data_name} not found for model {model_name}")
    
    def _check_thresholds(self, model_name: str, model_results: Dict, model_config: Dict):
        """
        Check metric results against thresholds and generate alerts.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        model_results : Dict
            Results for this model
        model_config : Dict
            Configuration for this model
        """
        # Get alert configuration
        alert_config = self.config.get('alerts', {})
        threshold_breach = alert_config.get('threshold_breach', True)
        
        if not threshold_breach:
            return
        
        # Get metrics configuration
        metrics_config = model_config.get('metrics', [])
        metric_configs = {m['name']: m for m in metrics_config if 'name' in m}
        
        # Get global thresholds
        global_thresholds = model_config.get('thresholds', {})
        
        # Check each metric result
        for metric_name, result_dict in model_results.items():
            # Skip metrics with errors
            if 'error' in result_dict:
                continue
                
            result = result_dict.get('result')
            if result is None:
                continue
            
            # Get threshold for this metric
            metric_config = metric_configs.get(metric_name, {})
            threshold = metric_config.get('threshold')
            
            # If no specific threshold, check global thresholds
            if threshold is None:
                threshold = global_thresholds.get(metric_name.lower())
            
            # Skip if no threshold defined
            if threshold is None:
                continue
                
            # Check threshold
            passed = result.passed if hasattr(result, 'passed') and result.passed is not None else None
            
            # If passed is explicitly set, use it
            if passed is not None:
                if not passed:
                    self._add_alert(model_name, metric_name, result.value, threshold, passed)
            # Otherwise compare value to threshold
            elif result.value is not None:
                # Determine comparison type based on metric
                metric_type = self._get_metric_type(metric_name)
                
                if metric_type == 'discrimination':
                    # For discrimination metrics, higher is usually better
                    passed = result.value >= threshold
                elif metric_type in ['calibration', 'stability']:
                    # For calibration and stability metrics, lower is usually better
                    passed = result.value <= threshold
                else:
                    # For unknown metrics, just use the threshold as is
                    passed = result.value <= threshold
                    
                if not passed:
                    self._add_alert(model_name, metric_name, result.value, threshold, passed)
    
    def _add_alert(self, model_name: str, metric_name: str, value: float, 
                  threshold: float, passed: bool):
        """
        Add an alert for a threshold breach.
        
        Parameters
        ----------
        model_name : str
            Name of the model
        metric_name : str
            Name of the metric
        value : float
            Value of the metric
        threshold : float
            Threshold for the metric
        passed : bool
            Whether the metric passed the threshold
        """
        alert = {
            'model': model_name,
            'metric': metric_name,
            'value': float(value),  # Convert to native float
            'threshold': float(threshold),  # Convert to native float
            'passed': bool(passed),  # Convert to native boolean
            'timestamp': self.timestamp
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert: {model_name} - {metric_name} = {value} (threshold: {threshold})")
    
    def _save_results(self):
        """Save results to disk."""
        # Create results directory if it doesn't exist
        results_path = os.path.join(self.results_dir, f"results_{self.timestamp}.json")
        
        # Convert results to JSON-serializable format
        json_results = self._results_to_json()
        
        # Save results using custom encoder for NumPy types
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, cls=NumpyEncoder)
            
        logger.info(f"Results saved to {results_path}")
        
        # Save alerts if any
        if self.alerts:
            alerts_path = os.path.join(self.results_dir, f"alerts_{self.timestamp}.json")
            with open(alerts_path, 'w') as f:
                json.dump(self.alerts, f, indent=2, cls=NumpyEncoder)
                
            logger.info(f"Alerts saved to {alerts_path}")
    
    def _results_to_json(self) -> Dict:
        """
        Convert results to JSON-serializable format.
        
        Returns
        -------
        Dict
            JSON-serializable results
        """
        json_results = {}
        
        for model_name, model_results in self.results.items():
            json_results[model_name] = {}
            
            for metric_name, result_dict in model_results.items():
                # Handle errors
                if 'error' in result_dict:
                    json_results[model_name][metric_name] = {
                        'error': result_dict['error'],
                        'timestamp': result_dict['timestamp']
                    }
                    continue
                
                # Get result
                result = result_dict.get('result')
                if result is None:
                    continue
                    
                # Convert MetricResult to dict with native Python types
                json_results[model_name][metric_name] = {
                    'value': float(result.value) if result.value is not None else None,
                    'passed': bool(result.passed) if result.passed is not None else None,
                    'threshold': float(result.threshold) if result.threshold is not None else None,
                    'details': self._convert_to_serializable(result.details),
                    'timestamp': result_dict['timestamp']
                }
                
                # Remove figure_data as it's not easily serializable
                if 'figure_data' in json_results[model_name][metric_name].get('details', {}):
                    del json_results[model_name][metric_name]['details']['figure_data']
        
        return json_results
        
    def _convert_to_serializable(self, obj):
        """
        Recursively convert NumPy types to native Python types.
        
        Parameters
        ----------
        obj : Any
            Object to convert
            
        Returns
        -------
        Any
            Converted object with only JSON-serializable types
        """
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._convert_to_serializable(obj.__dict__)
        else:
            return obj
    
    def _generate_reports(self):
        """Generate reports based on results."""
        # Get report configuration
        report_config = self.config.get('reporting', {})
        output_format = report_config.get('output_format', 'html')
        include_plots = report_config.get('include_plots', True)
        
        # For now, just generate a simple HTML report
        if output_format.lower() == 'html':
            self._generate_html_report(include_plots)
        elif output_format.lower() == 'json':
            # JSON report is already created by _save_results
            pass
        else:
            logger.warning(f"Unsupported report format: {output_format}")
    
    def _generate_html_report(self, include_plots: bool = True):
        """
        Generate an HTML report.
        
        Parameters
        ----------
        include_plots : bool
            Whether to include plots in the report
        """
        # Create report path
        report_path = os.path.join(self.report_dir, f"report_{self.timestamp}.html")
        
        # Basic HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CRMStudio Monitoring Report - {self.timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .alerts {{ background-color: #fff0f0; padding: 10px; border-left: 5px solid #ff0000; }}
                .metric-value {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>CRMStudio Monitoring Report</h1>
            <p>Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        """
        
        # Add alerts section if any
        if self.alerts:
            html += """
            <div class="alerts">
                <h2>Alerts</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Threshold</th>
                        <th>Status</th>
                    </tr>
            """
            
            for alert in self.alerts:
                status_class = "failed" if not alert['passed'] else "passed"
                status_text = "FAILED" if not alert['passed'] else "PASSED"
                
                html += f"""
                <tr>
                    <td>{alert['model']}</td>
                    <td>{alert['metric']}</td>
                    <td class="metric-value">{alert['value']:.4f}</td>
                    <td>{alert['threshold']:.4f}</td>
                    <td class="{status_class}">{status_text}</td>
                </tr>
                """
                
            html += """
                </table>
            </div>
            """
        
        # Add results by model
        for model_name, model_results in self.results.items():
            html += f"""
            <h2>Model: {model_name}</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Threshold</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
            """
            
            for metric_name, result_dict in model_results.items():
                # Handle errors
                if 'error' in result_dict:
                    html += f"""
                    <tr>
                        <td>{metric_name}</td>
                        <td colspan="3" class="failed">ERROR: {result_dict['error']}</td>
                        <td></td>
                    </tr>
                    """
                    continue
                
                # Get result
                result = result_dict.get('result')
                if result is None:
                    continue
                
                # Format value
                value_text = f"{result.value:.4f}" if result.value is not None else "N/A"
                
                # Format threshold
                threshold_text = f"{result.threshold:.4f}" if result.threshold is not None else "N/A"
                
                # Format status
                if result.passed is not None:
                    status_class = "passed" if result.passed else "failed"
                    status_text = "PASSED" if result.passed else "FAILED"
                else:
                    status_class = ""
                    status_text = "N/A"
                
                # Format details
                details_text = ""
                if result.details:
                    # Get up to 3 key metrics from details
                    keys = list(result.details.keys())[:3]
                    for key in keys:
                        if key != 'figure_data' and result.details[key] is not None:
                            value = result.details[key]
                            if isinstance(value, (int, float)):
                                value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
                            else:
                                value_str = str(value)
                            details_text += f"{key}: {value_str}<br>"
                
                html += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td class="metric-value">{value_text}</td>
                    <td>{threshold_text}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{details_text}</td>
                </tr>
                """
                
                # Add plot if available and plots are included
                if include_plots and result.has_figure():
                    try:
                        # Generate plot
                        metric_instance = self.METRIC_MAPPING.get(metric_name.lower())(model_name=model_name)
                        metric_instance.result = result
                        
                        # Save plot to file
                        plot_filename = f"{model_name}_{metric_name}_{self.timestamp}.png"
                        plot_path = os.path.join(self.report_dir, plot_filename)
                        metric_instance.save_plot(plot_path)
                        
                        # Add to HTML
                        html += f"""
                        <tr>
                            <td colspan="5" style="text-align: center;">
                                <img src="{plot_filename}" alt="{metric_name} plot" style="max-width: 100%;">
                            </td>
                        </tr>
                        """
                    except Exception as e:
                        logger.warning(f"Error generating plot for {metric_name}: {str(e)}")
            
            html += """
            </table>
            """
        
        # Close HTML
        html += """
        </body>
        </html>
        """
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(html)
            
        logger.info(f"HTML report saved to {report_path}")
