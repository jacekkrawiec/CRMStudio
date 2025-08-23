# src/crmstudio/core/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

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
        if not np.all(np.isfinite(y_pred)):
            raise ValueError("y_pred contains non-finite values, e.g. NaNs or INF.")
        if not np.all(np.isin(y_true, [0,1])): # I suppose at some point we'll need to support multi-class/continuous; let's leave it for now
            raise ValueError("y_true must be binary (0/1) for discrimination metrics.")
        return y_true, y_pred

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
        Compute metric and wrap into MetricResult/FigureResult object.
        
        """
        from tqdm import tqdm
        steps = ["Validating inputs", "Computing metric", "Wrapping results"]
        with tqdm(total = len(steps), desc = f"Computing {self.__class__.__name__}") as pbar:
            #validate inputs
            y_true, y_pred = self._validate_inputs(y_true, y_pred)
            pbar.update(1)

            #compute metric
            value, details = self._wrap_results(self._compute_raw(y_true, y_pred))
            pbar.update(1)

            #wrap results
            if self.metric_config.get("produce_figure",False):
                figure_data = details
                self.result = FigureResult(
                    name=helpers.pascal_to_snake(self.__class__.__name__),
                    figure_data = figure_data
                )
            else:
                threshold = self.metric_config.get("threshold")
                passed = (threshold is None) or (value >= threshold)
                
                self.result = MetricResult(
                    name=self.__class__.__name__,
                    value=value,
                    threshold=threshold,
                    passed=passed,
                    details = details
                )
            pbar.update(1)
        return self.result

    def _wrap_results(self, value, details: Dict[str, Any] = None) -> Union[float, Dict]:
        """
        Standardize result wrappiing into MetricResult or FigureResult.
        """
        # Handle different return types from _comput_raw (result)
        if isinstance(value, (int, float)):
            value, details = value, {}
        elif isinstance(value, dict):
            value, details = np.nan, value
        elif isinstance(value, (tuple, list)):
            value, *rest = value
            details = rest[0] if rest else {}
        else:
            raise ValueError("Unsupported return type from _compute_raw")
        return {"value": value, "details": details}



@dataclass
class MetricResult:
    name: str
    value: float
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)

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
    Standardizes curve data to consistent x,y coordinates regardless of the source metric.
    """

    def __init__(self, name, figure_data, details=Dict[str, Any]):
        """
        Parameters
        ----------
        name : str
            Name of the metric.
        figure_data : dict
            Data needed to render the figure. For curves, must contain coordinate data
            that can be mapped to x,y pairs.
        details : dict, optional
            Any extra information (e.g., p-value, n_obs, bins).
        """
        self.name = name
        self.details = details or {}
        
        # Standardize curve coordinates
        self.figure_data = self._standardize_coordinates(figure_data)
    
    def _standardize_coordinates(self, data: Dict) -> Dict:
        """
        Standardize different curve coordinate naming conventions to x,y pairs.
        
        Handles various input formats:
        - ROC curves: (fpr, tpr)
        - CAP curves: (x, y)
        - PR curves: (recall, precision)
        - Custom curves: Any pair that can be mapped to (x,y)
        
        Returns
        -------
        Dict
            Dictionary with standardized 'x' and 'y' keys, preserving original data
        """
        # Copy original data
        standardized = data.copy()
        
        # Known coordinate mappings
        coord_mappings = {
            # ROC curve coordinates
            ('fpr', 'tpr'): ('x', 'y'),
            # Precision-Recall curve coordinates
            ('recall', 'precision'): ('x', 'y'),
            # KS curve coordinates
            ('threshold', 'statistic'): ('x', 'y'),
            # Already standardized
            ('x', 'y'): ('x', 'y')
        }
        
        # Find matching coordinate pair
        for (src_x, src_y), (dst_x, dst_y) in coord_mappings.items():
            if src_x in data and src_y in data:
                if src_x != dst_x:
                    standardized[dst_x] = data[src_x]
                if src_y != dst_y:
                    standardized[dst_y] = data[src_y]
                break
        
        return standardized

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
    

class CurveMetric(BaseMetric):
    """
    Base class for metrics based on curves (e.g. ROC, CAP, PR, KS).
    """

    def __init__(self, model_name: str, config: Dict = None, config_path: str = None, **kwargs):
        super().__init__(model_name, config, config_path, **kwargs)
        # Move plt.rcParams update to initialization
        self._style = self._init_style()

    def _init_style(self):
        """
        Initialize style configuration for plotting. 
        Can be overwritten by user-procided styles.
        By default, uses template from config_loader
        """
        default_style_path = "src/crmstudio/core/templates/figure_style.yaml"
        style = load_config(default_style_path)
        return style

    def _compute_raw(self, y_true, y_pred, **kwargs):
        return super()._compute_raw(y_true, y_pred, **kwargs)

    def _compute_auc(self, x, y) -> float:
        """
        Compute Area Under Curve (AUC) using trapezoidal rule.
        This method assumes x and y represent the curve coordinates.
        TODO: sort x and y if not sorted
        """
        return np.trapz(y, x)
    
    def _compute_ks(self, x, y) -> Tuple[float, float]:
        """
        Compute Kolmogorov-Smirnov (KS) statistic.
        Returns KS value and corresponding x (threshold).
        """
        ks_statistic = np.max(np.abs(y - x))
        ks_index = np.argmax(np.abs(y - x))
        ks_threshold = x[ks_index]
        return ks_statistic, ks_threshold
    
    def _plot_curve(self, figure_data: Dict, style: Optional[Dict] = None) -> Dict:
        """
        Plot the curve using matplotlib and return figure data.
        Uses standardized x,y coordinates from FigureResult.
        
        Parameters
        ----------
        figure_data : Dict
            Dictionary containing at minimum 'x' and 'y' coordinates
        style : Dict, optional
            Style configuration dictionary; if no style is provided, default template is used.
        
        Returns
        -------
        Dict
            Dictionary containing base64 encoded image and dimensions

        -------
        TODO: Consider using plotly for interactive plots.
        TODO: Consider moving plotting to separate utility module.
        """
        if style is None:
            style = self._style

        palette = style["colors"]["palette"]
        x = figure_data['x']
        y = figure_data['y']
        
        # Calculate score (AUC) if not provided
        score = figure_data.get('score', self._compute_auc(x, y))
        
        fig, ax = plt.subplots()
        ax.plot(x, y, color=palette[0], lw=2, label=f"Curve (AUC = {score:.2f})")
        
        if style.get("show_diagonal", True):
            ax.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--', label='Random')
        
        ax.legend()
        ax.grid(True)
        
        # Convert matplotlib figure to a dictionary representation
        from io import BytesIO
        import base64
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)  # Clean up the figure
        
        return {
            "image_base64": image_base64,
            "width": fig.get_size_inches()[0] * fig.dpi,
            "height": fig.get_size_inches()[1] * fig.dpi
        }
    
    def show_plot(self, figure_data: Dict, style: Optional[Dict]=  None):
        """ shows plot """
        image_base64 = self._plot_curve(figure_data, style)['image_base64']
        from io import BytesIO
        import base64
        img_data = base64.b64decode(image_base64)
        img = plt.imread(BytesIO(img_data))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def save_plot(self, figure_data: Dict, filepath: str):
        """
        Save the plot to disk if needed.
        
        Parameters
        ----------
        figure_data : Dict
            Dictionary containing base64 encoded image
        filepath : str
            Path where to save the plot
        """
        import base64
        img_data = base64.b64decode(figure_data["image_base64"])
        with open(filepath, "wb") as f:
            f.write(img_data)