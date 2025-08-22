# src/crmstudio/core/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import re

from .config_loader import load_config
from ..utils import helpers

class BaseMetric(ABC):
    """
    Abstract base class for all monitoring metrics.
    All metrics (PD, LGD, EAD) should inherit from this class.
    Thresholds and other parameters are taken from config.
    """

    def __init__(self, model_name: str, config: Dict =None, config_path: str = None, **kwargs):
        """
        Parameters
        ----------
        name : str
            Name of the metric.
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
        self.metric_config = self._extract_metric_config(**kwargs)
        self.result = None

    def _extract_metric_config(self, **kwargs) -> dict:
        """
        Extract metric-specific configuration from loaded YAML or from kwargs overrides.
        """
        class_pascal = self.__class__.__name__
        class_snake = helpers.pascal_to_snake(class_pascal)
        class_names = [class_pascal, class_snake]
        class_names = [class_pascal, class_snake]
        metrics = self.config.get("models", {}).get(self.model_name, {}).get("metrics", [])
        metrics_cfg = next((m for m in metrics if m["name"].lower() in class_names), {})
        return {**metrics_cfg, **kwargs}

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

    def compute(self, y_true, y_pred, **kwargs):
        """
        Compute metric and wrap into MetricResult object.
        """
        if self.metric_config.get("produce_figure",False):
            figure_data = self._compute_raw(y_true, y_pred)
            return FigureResult(
                name=helpers.pascal_to_snake(self.__class__.__name__),
                figure_data = figure_data,
                details = {}
            )
        # Merge config kwargs with compute kwargs
        value = self._compute_raw(y_true, y_pred)
        if len(value) > 1: #this is wrong implementation, there can ve dictionary passed to value and will be len>1 but [1] won't work
            details = value[1]
            value = value[0]
        else:
            details = {}
        threshold = self.metric_config.get("threshold")
        passed = (threshold is None) or (value >= threshold)
        self.result = MetricResult(
            name=self.__class__.__name__,
            value=value,
            threshold=threshold,
            passed=passed,
            details = details
        )
        return self.result

class MetricResult:
    """
    Encapsulates the result of a single metric calculation.
    Provides standardized output for reporting.
    """

    def __init__(self, name, value, threshold=None, passed=None, details=Dict[str, Any]):
        """
        Parameters
        ----------
        name : str
            Name of the metric.
        value : float
            Calculated metric value.
        threshold : float, optional
            Threshold value used for pass/fail evaluation.
        passed : bool, optional
            Indicates whether metric passed threshold.
        additional_info : dict, optional
            Any extra information (e.g., p-value, n_obs, bins).
        """
        self.name = name
        self.value = value
        self.threshold = threshold
        self.passed = passed
        self.details = details or {}

    def summary(self):
        """
        Returns a human-readable summary string.
        """
        status = "PASSED" if self.passed else "FAILED"
        threshold_str = f"{self.threshold}" if self.threshold is not None else "N/A"
        return f"{self.name}: {self.value:.4f} | Threshold: {threshold_str} | {status}"

    def to_dict(self):
        """
        Returns standardized dictionary representation of the result.
        Suitable for reporting to Excel, JSON, or dashboards.
        """
        return {
            "metric": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            **self.details
        }

    def __repr__(self):
        return f"<MetricResult {self.name}: {self.value:.4f}, passed={self.passed}>"

class FigureResult:
    """
    Encapsulates the result of a metric calculation that produces a figure.
    """

    def __init__(self, name, figure_data, details=Dict[str, Any]):
        """
        Parameters
        ----------
        name : str
            Name of the metric.
        figure_data : Any
            Data needed to render the figure (e.g., plotly figure dict).
        additional_info : dict, optional
            Any extra information (e.g., p-value, n_obs, bins).
        """
        self.name = name
        self.figure_data = figure_data
        self.details = details or {}

    def to_dict(self):
        """
        Returns standardized dictionary representation of the figure result.
        Suitable for reporting to dashboards.
        """
        return {
            "metric": self.name,
            "figure_data": self.figure_data,
            **self.details
        }

    def __repr__(self):
        return f"<FigureResult {self.name}>"