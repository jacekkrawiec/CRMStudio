# src/crmstudio/core/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


from .config_loader import load_config
from ..utils import helpers
from .data_classes import MetricResult
from .plotting import PlottingService

class SubpopulationAnalysisMixin:
    """Mixin for analyzing metrics across different subpopulations."""
    
    def _compute_by_group(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         group_index: np.ndarray,
                         group_metadata: Dict = None) -> MetricResult:
        """Core implementation of subpopulation analysis."""
        metadata = group_metadata or {}
        group_type = metadata.get('type', 'custom')
        group_name = metadata.get('name', 'group')
        
        results = []
        unique_groups = np.sort(np.unique(group_index)) if group_type == 'time' else np.unique(group_index)
        
        for group in unique_groups:
            mask = group_index == group
            n_obs = np.sum(mask)
            
            if n_obs > 0:
                result = self.compute(
                    y_true=y_true[mask],
                    y_pred=y_pred[mask]
                )
                results.append({
                    'group': group,
                    'group_type': group_type,
                    'group_name': group_name,
                    'n_obs': n_obs,
                    'n_defaults': int(np.sum(y_true[mask])),
                    'value': result.value,
                    'passed': result.passed,
                    'figure_data': result.figure_data,
                    **result.details
                })
                
        # Create DataFrame from results
        results_df = pd.DataFrame(results)
        
        # Return a MetricResult object with the DataFrame in details
        return MetricResult(
            name=f"{self.__class__.__name__}ByGroup",
            details={
                'group_type': group_type,
                'group_name': group_name,
                'n_groups': len(results)
            },
            figure_data={
                'results_df': results_df
            }
        )

    # User-friendly wrapper methods
    def compute_over_time(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     time_index: np.ndarray, freq: str = 'Y') -> MetricResult:
        """
        Compute metric over time periods.
        
        Parameters
        ----------
        time_index : array-like
            Dates or time periods for each observation
        freq : str
            Frequency for aggregation ('M'=monthly, 'Q'=quarterly, 'Y'=yearly)
        
        Returns
        -------
        MetricResult
            Metric result object containing the group analysis results
        """
        if freq not in ['M', 'Q', 'Y']:
            raise ValueError("Frequency must be one of: 'M', 'Q', 'Y'")
            
        # Convert time_index to pandas datetime and period
        # time_index = pd.to_datetime(time_index)
        period_index = pd.Series(time_index).dt.to_period(freq).values
        
        return self._compute_by_group(
            y_true=y_true,
            y_pred=y_pred,
            group_index=period_index,
            group_metadata={
                'type': 'time',
                'name': 'period'
            }
        )

    def compute_by_segment(self, y_true: np.ndarray, y_pred: np.ndarray,
                          segments: np.ndarray, segment_labels: Dict = None) -> MetricResult:
        """
        Compute metric for different segments.
        
        Parameters
        ----------
        segments : array-like
            Segment identifier for each observation
        segment_labels : dict, optional
            Mapping of segment IDs to display labels
            
        Returns
        -------
        MetricResult
            Metric result object containing the group analysis results
        """
        return self._compute_by_group(
            y_true=y_true,
            y_pred=y_pred,
            group_index=segments,
            group_metadata={
                'type': 'segment',
                'name': 'calibration_segment',
                'labels': segment_labels
            }
        )

    def compute_by_range(self, y_true: np.ndarray, y_pred: np.ndarray,
                        values: np.ndarray, bins: Union[int, list] = 10,
                        labels: list = None) -> MetricResult:
        """
        Compute metric across value ranges (e.g., LTV ranges).
        
        Parameters
        ----------
        values : array-like
            Values to bin (e.g., LTV values)
        bins : int or list
            Number of bins or custom bin edges
        labels : list, optional
            Custom labels for bins
            
        Returns
        -------
        MetricResult
            Metric result object containing the group analysis results
        """
        if isinstance(bins, int):
            group_index = pd.qcut(values, q=bins, labels=labels)
        else:
            group_index = pd.cut(values, bins=bins, labels=labels)
            
        return self._compute_by_group(
            y_true=y_true,
            y_pred=y_pred,
            group_index=group_index,
            group_metadata={'type': 'range', 'name': 'value_range'}
        )

class BaseMetric(ABC, SubpopulationAnalysisMixin):
    """
    Abstract base class for all monitoring metrics.
    All metrics (PD, LGD, EAD) should inherit from this class.
    Thresholds and other parameters are taken from config.
    """

    def __init__(self, model_name: str, metric_type: str = "unknown", config: Dict = None, config_path: str = None, **kwargs):
        """
        Parameters
        ----------
        model_name : str
            Name of the metric.
        metric_type : str, optional
            Type of metric ('curve', 'distribution', or custom)
        config : dict, optional
            Configuration dict containing parameters, e.g.:
            {"threshold": 0.6, "bins": 10, ...}
        config_path : str, optional
            Path to a configuration file (YAML/JSON) to load parameters from.
            If both config and config_path are provided, config takes precedence.
        **kwargs : additional keyword arguments
            Additional parameters to override config settings for single metric run.
        """
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)
        self.model_name = model_name
        self._metric_type = metric_type
        self.metric_config = self._extract_metric_config(**kwargs)
        self.result = None
        self._plotting_service = None
        self._style = self._init_style()

    def _extract_metric_config(self, **kwargs) -> dict:
        """
        Extract metric-specific configuration from loaded YAML or from kwargs overrides.
        """
        class_pascal = self.__class__.__name__
        class_snake = helpers.pascal_to_snake(class_pascal)
        class_names = [class_pascal, class_snake]
        metrics = self.config.get("models", {}).get(self.model_name, {}).get("metrics", [])
        metrics_cfg = next((m for m in metrics if m["name"].lower() in class_names), {})
        return {**metrics_cfg, **kwargs}

    def _get_param(self, param_name: str, default: Any = None) -> Any:
        """
        Get parameter with validation.
        """
        value = self.metric_config.get("params", {}).get(param_name, default)
        if value is None:
            raise ValueError(f"Required parameter '{param_name}' not found in metric configuration.")
        return value

    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standard input validation.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        # below commented out as it assumes only PD/binary use
        # if not np.all(np.isfinite(y_pred)):
        #     raise ValueError("y_pred contains non-finite values, e.g. NaNs or INF.")
        # if not np.all(np.isin(y_true, [0,1])): # I suppose at some point we'll need to support multi-class/continuous; let's leave it for now
        #     raise ValueError("y_true must be binary (0/1) for discrimination metrics.")
        return y_true, y_pred
        
    @property
    def plotting_service(self) -> PlottingService:
        """
        Get the plotting service instance.
        
        Returns
        -------
        PlottingService
            The plotting service instance
        """
        if self._plotting_service is None:
            self._plotting_service = PlottingService()
        return self._plotting_service

    @abstractmethod
    def _compute_raw(self, y_true, y_pred, **kwargs):
        """
        Compute the raw metric value.

        Returns
        -------
        tuple
            (value, additional_info)
        """
        pass

    def compute(self, **kwargs):
        """
        Computes the metric using provided keyword arguments and wraps the result into a MetricResult.
        This method performs the following steps:
            1. Validates the input arrays (`y_true` and `y_pred`) using the internal `_validate_inputs` method.
            2. Computes the metric by calling the internal `_compute_raw` method, which should return a dictionary-like result.
            3. Optionally applies a threshold from `self.metric_config` to the result, updating the `threshold` and `passed` attributes if applicable.
            4. Updates the progress bar at each step for user feedback.
        Parameters:
            **kwargs: Arbitrary keyword arguments. Must include:
                - y_true: Ground truth target values.
                - y_pred: Estimated target values.
        Returns:
            MetricResult: The computed and wrapped metric result object.
        Raises:
            KeyError: If required keys ('y_true', 'y_pred') are not present in kwargs.
            Exception: Propagates exceptions from validation or computation steps.
        """
        #validate inputs
        if 'y_true' in kwargs and 'y_pred' in kwargs:
            y_true = kwargs.get('y_true')
            y_pred = kwargs.get('y_pred')
            y_true, y_pred = self._validate_inputs(y_true, y_pred)

            #compute metric
            # Remove 'y_true' and 'y_pred' from kwargs before passing to _compute_raw
            compute_kwargs = {k: v for k, v in kwargs.items() if k not in ('y_true', 'y_pred')}
            result = self._compute_raw(y_true=y_true, y_pred=y_pred, **compute_kwargs)  # note it's always a MetricResult object
        else:
            result = self._compute_raw(**kwargs)

        #apply threshold if exists
        if result.value is not None and result.threshold is None:
            # Try to get threshold from different possible locations
            threshold = self.metric_config.get('threshold')
            if threshold is None and 'params' in self.metric_config:
                threshold = self.metric_config.get('params', {}).get('threshold')
            
            if threshold is not None:
                result.threshold = threshold
                result.passed = result.value >= threshold

        self.result = result
        return self.result
    
    @staticmethod
    def _init_style():
        """
        Initialize style configuration for plotting. 
        Can be overwritten by user-procided styles.
        By default, uses template from config_loader
        """
        pkg_dir = Path(__file__).parent
        templates_dir = pkg_dir / "templates"
        default_style_path = templates_dir / "figure_style.yaml"
        if default_style_path.exists():
            return load_config(str(default_style_path))
        return load_config("src/crmstudio/core/templates/figure_style.yaml")
    
    def _apply_style(self, style: Optional[Dict] = None):
        """
        Apply the given style to the plotting service.
        
        Uses the plotting_service property to ensure the service is initialized.
        
        Parameters
        ----------
        style : Dict, optional
            Style configuration to apply
            
        Returns
        -------
        PlottingService
            The configured plotting service
        """
        service = self.plotting_service  # Uses property which initializes if needed
        if style is not None:
            service.set_style(style)
        elif self._style is not None:
            service.set_style(self._style)
        return service
    
    def show_plot(self, results: Optional[Union[MetricResult, Dict]] = None, style: Optional[Dict] = None):
        """
        Show the plot with the given results and style.
        
        This is the main entry point for plotting in CRMStudio.
        
        Parameters
        ----------
        results : MetricResult or Dict, optional
            The metric results to plot. If None, uses self.result
        style : Dict, optional
            Style configuration to use
        """
        # If no results provided, use the stored result
        if results is None:
            if self.result is None:
                raise ValueError("No results to plot. Run compute() first or provide results.")
            results = self.result
        
        # Convert dictionary to MetricResult if needed
        if isinstance(results, dict) and not isinstance(results, MetricResult):
            results = MetricResult(name=self.__class__.__name__, figure_data=results)
        
        # Apply style to the plotting service
        service = self._apply_style(style)
        
        # Let the plotting service handle the appropriate type of plot
        plot_data = service.plot(results, plot_type=self.metric_type)
        
        # Display the image
        service.display_image(plot_data)

    def save_plot(self, filepath: str, results: Optional[Union[MetricResult, Dict]] = None, style: Optional[Dict] = None):
        """
        Save the plot to disk.
        
        This is the main entry point for saving plots in CRMStudio.
        
        Parameters
        ----------
        filepath : str
            Path where to save the plot
        results : MetricResult or Dict, optional
            The metric results to plot. If None, uses self.result
        style : Dict, optional
            Style configuration to use
        """
        # If no results provided, use the stored result
        if results is None:
            if self.result is None:
                raise ValueError("No results to plot. Run compute() first or provide results.")
            results = self.result
        
        # Convert dictionary to MetricResult if needed
        if isinstance(results, dict) and not isinstance(results, MetricResult):
            results = MetricResult(name=self.__class__.__name__, figure_data=results)
        
        # Apply style to the plotting service
        service = self._apply_style(style)
        
        # Let the plotting service handle the appropriate type of plot
        plot_data = service.plot(results, plot_type=self.metric_type)
        
        # Save the image
        service.save_image(plot_data, filepath)

    @property
    def metric_type(self):
        """Return the type of the metric."""
        return self._metric_type



