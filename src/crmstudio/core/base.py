# src/crmstudio/core/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from io import BytesIO
import base64
import pandas as pd

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

    def compute(self, **kwargs):
        """
        Compute metric and wrap into MetricResult/FigureResult object.
        
        """
        from tqdm import tqdm
        steps = ["Validating inputs", "Computing metric", "Wrapping results"]
        with tqdm(total = len(steps), desc = f"Computing {self.__class__.__name__}") as pbar:
            #validate inputs
            if 'y_true' in kwargs and 'y_pred' in kwargs:
                y_true = kwargs.get('y_true')
                y_pred = kwargs.get('y_pred')
                y_true, y_pred = self._validate_inputs(y_true, y_pred)
            
                pbar.update(1)

                #compute metric
                wrapped_result = self._wrap_results(self._compute_raw(y_true = y_true, y_pred = y_pred)) #note it's always a dictionary
                pbar.update(1)
            elif 'fr' in kwargs and kwargs.get('fr') is not None:
                wrapped_result = self._wrap_results(self._compute_raw(fr = kwargs.get('fr')))
                pbar.update(2)
            #wrap results
            if self.metric_config.get("produce_figure",False):
                figure_data = wrapped_result['details']
                self.result = FigureResult(
                    name=helpers.pascal_to_snake(self.__class__.__name__),
                    figure_data = figure_data
                )
            else:
                threshold = self.metric_config.get("threshold")
                value = wrapped_result['value']
                details = wrapped_result['details']
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
            details = value
            value = np.nan
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
        return f"{self.name}: {self.value:.6f} | Threshold: {threshold_str} | {status}"

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

    def __init__(self, name, figure_data, details: Optional[Dict[str, Any]] = None):
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
        standardized = data
        
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

class SubpopulationAnalysisMixin:
    """Mixin for analyzing metrics across different subpopulations."""
    
    def _compute_by_group(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         group_index: np.ndarray,
                         group_metadata: Dict = None) -> pd.DataFrame:
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
                if isinstance(result, MetricResult):
                    results.append({
                        'group': group,
                        'group_type': group_type,
                        'group_name': group_name,
                        'n_obs': n_obs,
                        'n_defaults': int(np.sum(y_true[mask])),
                        'value': result.value,
                        'passed': result.passed,
                        **result.details
                    })
                elif isinstance(result, FigureResult):
                    # Handle FigureResult
                    results.append({
                        'group': group,
                        'group_type': group_type,
                        'group_name': group_name,
                        'n_obs': n_obs,
                        'n_defaults': int(np.sum(y_true[mask])),
                        'figure_data': result.figure_data,
                        **result.details
                    })

        return pd.DataFrame(results)

    # User-friendly wrapper methods
    def compute_over_time(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     time_index: np.ndarray, freq: str = 'Y') -> pd.DataFrame:
        """
        Compute metric over time periods.
        
        Parameters
        ----------
        time_index : array-like
            Dates or time periods for each observation
        freq : str
            Frequency for aggregation ('M'=monthly, 'Q'=quarterly, 'Y'=yearly)
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
                          segments: np.ndarray, segment_labels: Dict = None) -> pd.DataFrame:
        """
        Compute metric for different segments.
        
        Parameters
        ----------
        segments : array-like
            Segment identifier for each observation
        segment_labels : dict, optional
            Mapping of segment IDs to display labels
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
                        labels: list = None) -> pd.DataFrame:
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

    def _plot_group_results(self, results: pd.DataFrame, style: Optional[Dict] = None) -> Dict:
        """
        Plot metric results across groups.
        
        Parameters
        ----------
        results : pd.DataFrame
            DataFrame returned by compute_over_time/compute_by_segment/compute_by_range
        style : Dict, optional
            Style configuration dictionary
            
        Returns
        -------
        Dict
            Dictionary containing plot image data and dimensions
        """
        if style is None:
            if not hasattr(self, '_style'):
                self._style = self._init_style()
            style = self._style
        
        # Check if figure data in results
        has_figure_data = 'figure_data' in results.columns
        # if not has_figure_data:
        #     raise ValueError("No figure data available in results for plotting.")

        # Extract style configurations
        colors = style.get('colors', {})
        fig_style = style.get('figure', {})
        grid_style = style.get('grid', {})
        line_style = style.get('lines', {})
        fig, ax = plt.subplots(figsize=fig_style.get('size', [10, 6]))
        if has_figure_data:
            if isinstance(self, CurveMetric):
                # Plot curves for each group
                
                for _, row in results.iterrows():
                    group = row['group']
                    figure_data = row['figure_data']
                    ax.plot(figure_data['x'], figure_data['y'], label=f"{group}", 
                            alpha = line_style.get('main_alpha', 0.7))
                #set labels from first figure_data
                first_figure = results.iloc[0]['figure_data']
                ax.set_xlabel(first_figure.get('xlabel', 'X'))
                ax.set_ylabel(first_figure.get('ylabel', 'Y'))

                # Add reference line if applicable
                if hasattr(self, '_add_reference_line'):
                    self._add_reference_line(ax, results)
            elif isinstance(self, DistributionAssociationMetric):
                # For distribution metrics create subplots
                nrows = (len(results)+1) // 2
                fig, axes = plt.subplots(
                    nrows = nrows,
                    ncols = 2,
                    figsize = (15, 5 * nrows)
                )
                axes = axes.flatten() #this is new -> earlier it wasn't flat and that's why we modified idx for indexing - see below
                
                for idx, (_, row) in enumerate(results.iterrows()):
                    self._plot_figure(
                        ax = axes[idx],
                        figure_data=row['figure_data'],
                        style=style,
                        group_label=str(row['group'])
                    )
        else:
            group_type = results['group_type'].iloc[0]
        
            if group_type == 'time':
                # Time series plot
                x_values = [str(p) for p in results['group']]
                ax.plot(x_values, results['value'],
                    marker='o',
                    color=colors.get('main', '#1f77b4'),
                    linewidth=line_style.get('main_width', 2))
                plt.xticks(rotation=45)
                ax.set_xlabel('Period')
            else:
                # Bar plot
                x = range(len(results))
                ax.bar(x, results['value'],
                    color=colors.get('main', '#1f77b4'),
                    alpha=0.7)
                
                if 'ci_lower' in results.columns:
                    ax.errorbar(x, results['value'],
                            yerr=[results['value'] - results['ci_lower'],
                                    results['ci_upper'] - results['value']],
                            fmt='none',
                            color='black',
                            capsize=5)
                    
                plt.xticks(x, results['group'], rotation=45)
                ax.set_xlabel(results['group_name'].iloc[0].title())
            
            ax.set_ylabel(f"{self.__class__.__name__} Value")
            
            # Add sample size as secondary axis
            ax2 = ax.twinx()
            ax2.plot(range(len(results)), results['n_obs'],
                    color='gray',
                    linestyle=':',
                    alpha=0.6,
                    label='Sample Size')
            ax2.set_ylabel('Sample Size')
        
        # Common styling
        if grid_style.get('show', True):
            if has_figure_data and isinstance(self, DistributionAssociationMetric):
                for idx, _ in enumerate(axes):
                    ax = axes[idx]
                    ax.grid(True,
                        linestyle=grid_style.get('linestyle', '--'),
                        alpha=grid_style.get('alpha', 0.3))
            else:
                ax.grid(True,
                    linestyle=grid_style.get('linestyle', '--'),
                    alpha=grid_style.get('alpha', 0.3))
                
        # Convert to image data
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return {
            "image_base64": image_base64,
            "width": fig.get_size_inches()[0] * fig.dpi,
            "height": fig.get_size_inches()[1] * fig.dpi
        }
    
    def show_group_plot(self, results: pd.DataFrame, style: Optional[Dict] = None):
        """Display the group analysis plot."""
        plot_data = self._plot_group_results(results, style)
        img_data = base64.b64decode(plot_data["image_base64"])
        img = plt.imread(BytesIO(img_data))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def save_group_plot(self, results: pd.DataFrame, filepath: str, style: Optional[Dict] = None):
        """Save the group analysis plot to file."""
        plot_data = self._plot_group_results(results, style)
        img_data = base64.b64decode(plot_data["image_base64"])
        with open(filepath, "wb") as f:
            f.write(img_data)

class CurveMetric(BaseMetric, SubpopulationAnalysisMixin):
    """Base class for metrics based on curves (e.g. ROC, CAP, PR, KS)."""

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
        else:
            self._style = style

        # Extract style configurations
        colors = style.get('colors', {})
        fig_style = style.get('figure', {})
        grid_style = style.get('grid', {})
        line_style = style.get('lines', {})
        legend_style = style.get('legend', {})

        # Create figure with consistent size
        fig_size = fig_style.get('size', [8, 5])
        fig, ax = plt.subplots(figsize=fig_size)

        # Extract curve data and metadata
        x = figure_data['x']
        y = figure_data['y']
        title = figure_data.get('title', 'Model Performance')
        xlabel = figure_data.get('xlabel', 'Score')
        ylabel = figure_data.get('ylabel', 'Performance')
        
        # Plot main curve with consistent style
        ax.plot(x, y, 
               color=colors.get('palette', ['#1f77b4'])[0],
               alpha=line_style.get('main_alpha', 0.8),
               linewidth=line_style.get('main_width', 2),
               label='Model')
        
        # Add reference line if needed
        if style.get('axes', {}).get('show_diagonal', True):
            ax.plot([0, 1], [0, 1], 
                   color=colors.get('reference_line', '#ff7f0e'),
                   linestyle='--',
                   alpha=line_style.get('reference_alpha', 0.5),
                   linewidth=line_style.get('reference_width', 1.5),
                   label='Random')

        # Apply common styling
        if grid_style.get('show', True):
            ax.grid(True,
                   linestyle=grid_style.get('linestyle', '--'),
                   alpha=grid_style.get('alpha', 0.3),
                   color=grid_style.get('color', '#cccccc'))

        # Set axis labels with consistent font sizes
        ax.set_xlabel(xlabel, fontsize=fig_style.get('label_fontsize', 10))
        ax.set_ylabel(ylabel, fontsize=fig_style.get('label_fontsize', 10))
        ax.set_title(title, fontsize=fig_style.get('title_fontsize', 12))
        
        # Set axis limits if provided
        if 'xlim' in figure_data:
            ax.set_xlim(figure_data['xlim'])
        if 'ylim' in figure_data:
            ax.set_ylim(figure_data['ylim'])
            
        # Add percentage ticks if specified
        if figure_data.get('use_percentage_ticks', False):
            step = style.get('axes', {}).get('percentages', {}).get('step', 20)
            ax.set_xticks(np.arange(0, 101, step))
            ax.set_xticklabels([f"{x}%" for x in range(0, 101, step)])
            if figure_data.get('use_percentage_ticks_y', False):
                ax.set_yticks(np.arange(0, 101, step))
                ax.set_yticklabels([f"{y}%" for y in range(0, 101, step)])

        # Configure legend with consistent style
        ax.legend(fontsize=legend_style.get('fontsize', 10),
                 framealpha=legend_style.get('framealpha', 0.8))

        # Set tick label sizes
        ax.tick_params(labelsize=fig_style.get('tick_fontsize', 8))
        
        # Convert matplotlib figure to a dictionary representation
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
        img_data = base64.b64decode(image_base64)
        img = plt.imread(BytesIO(img_data))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def save_plot(self, figure_data: Dict, filepath: str, style: Optional[Dict]=  None):
        """
        Save the plot to disk if needed.
        
        Parameters
        ----------
        figure_data : Dict
            Dictionary containing base64 encoded image
        filepath : str
            Path where to save the plot
        """
        plot_data = self._plot_curve(figure_data, style)
        img_data = base64.b64decode(plot_data["image_base64"])
        with open(filepath, "wb") as f:
            f.write(img_data)


class DistributionAssociationMetric(BaseMetric, SubpopulationAnalysisMixin):
    """Base class for metrics based on distribution comparisons."""
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
        """
        Compute the raw metric value for distribution/association metrics.

        Returns
        -------
        tuple
            (value, additional_info)
        """
        pass

    def _plot_figure(self, ax, figure_data: Dict, style: Dict, group_label: str = None):
        """
        Unified plotting method for all distribution-based visualizations.

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes on which to plot the figure.
        figure_data: Dict
            The data to be plotted with keys:
            - 'x': x-coordinates for the plot
            - 'y': y-coordinates for the plot
            - 'xlabel': label for the x-axis
            - 'ylabel': label for the y-axis
            - 'title': title of the plot
            - 'x_ref': x-coordinates for the reference line (optional)
            - 'y_ref': y-coordinates for the reference line (optional)
            - 'x_def': x-coordinates for the defaulted population (optional)
            - 'y_def': y-coordinates for the defaulted population (optional)
            - 'x_ndef': x-coordinates for the non-defaulted population (optional)
            - 'y_ndef': y-coordinates for the non-defaulted population (optional)
            - 'n_obs': number of observations
            - 'value': metric value (optional)
        style: Dict
            The style configuration for the plot.
        group_label: str, optional
            The label for the group being plotted.
        """
        colors = style.get('colors', {})
        line_style = style.get('lines', {})

        # Plot distributions if present (histogram case)
        if all(k in figure_data for k in ['x_def', 'y_def', 'x_ndef', 'y_ndef']):
            # ax.bar(figure_data['x_def'], figure_data['y_def'], color=colors.get('defaulted', 'red'), label='Defaulted', alpha=0.25)
            # ax.bar(figure_data['x_ndef'], figure_data['y_ndef'], color=colors.get('non_defaulted', 'green'), label='Non-defaulted', alpha=0.25)
            ax.hist(figure_data['x_def'], bins=figure_data['bin_edges'], weights=figure_data['y_def'],
                alpha=0.5, color=colors.get('defaulted', 'red'),
                label='Defaulted')
            ax.hist(figure_data['x_ndef'], bins=figure_data['bin_edges'], weights=figure_data['y_ndef'],
                alpha=0.5, color=colors.get('non_defaulted', 'green'),
                label='Non-defaulted')
        # Plot main curve if present (KS, Lift, Gain case)
        if 'x' in figure_data and 'y' in figure_data:
            if "actual_label" in figure_data:
                label = figure_data["actual_label"]
            else:
                label = "Actual"
            ax.plot(figure_data['x'], figure_data['y'],
                    color=colors.get('main', '#1f77b4'),
                    linewidth=line_style.get('main_width', 2),
                    label=label)
        
        # Plot reference curve if present
        if 'x_ref' in figure_data and 'y_ref' in figure_data:
            if "ref_label" in figure_data:
                label = figure_data["ref_label"]
            else:
                label = "Reference"
            ax.plot(figure_data['x_ref'], figure_data['y_ref'],
                   color=colors.get('reference', '#ff7f0e'),
                   linestyle='--',
                   alpha=line_style.get('reference_alpha', 0.5),
                   label=label)
        
        # Set labels
        ax.set_xlabel(figure_data.get('xlabel', 'X'))
        ax.set_ylabel(figure_data.get('ylabel', 'Y'))

        # Set title with group label and sample size
        title = []
        if 'title' in figure_data:
            title.append(figure_data['title'])
        if group_label:
            title.append(f"{group_label}")
        if 'n_obs' in figure_data:
            title.append(f"(n={figure_data['n_obs']})")
        if title:
            ax.set_title(" ".join(title))

        # Add metric value if present
        if 'value' in figure_data:
            ax.text(0.95, 0.95, f"Value: {figure_data['value']:.3f}",
                   transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='bottom')
        
        # Add grid if specified in style
        grid_style = style.get('grid', {})
        if grid_style.get('show', True):
            ax.grid(True, linestyle=grid_style.get('linestyle', '--'), alpha=grid_style.get('alpha', 0.3))

        ax.legend()

    def _plot_histogram(self, ax, figure_data: Dict, style: Dict, group_label: str = None):
        """Plot histogram for defaulted/non-defaulted populations."""
        colors = style.get('colors', {})

        bin_edges = figure_data['bin_edges']

        # Plot histograms
        ax.hist(bin_edges[:-1], bins=bin_edges, weights=figure_data['hist_defaulted'],
               alpha=0.5, color=colors.get('defaulted', 'red'),
               label='Defaulted')
        ax.hist(bin_edges[:-1], bins=bin_edges, weights=figure_data['hist_non_defaulted'],
               alpha=0.5, color=colors.get('non_defaulted', 'green'),
               label='Non-defaulted')
        
        # Add labels and title
        ax.set_xlabel(figure_data.get('xlabel', 'Score'))
        ax.set_ylabel(figure_data.get('ylabel', 'Frequency'))
        if group_label:
            ax.set_title(f"{group_label} (n={figure_data.get('n_obs', '')})")
        
        ax.legend()
        
    def _plot_distribution(self, ax, figure_data: Dict, style: Dict, group_label: str = None):
        """Plot distribution curves (KS, Lift, Gain)."""
        
        colors = style.get('colors', {})
        line_style = style.get('lines', {})
        
        # Plot main curves
        ax.plot(figure_data['x'], figure_data['y'],
                color=colors.get('main', '#1f77b4'),
                linewidth=line_style.get('main_width', 2),
                label='Actual')
        
        # Add reference line if present
        if 'x_ref' in figure_data and 'y_ref' in figure_data:
            ax.plot(figure_data['x_ref'], figure_data['y_ref'],
                   color=colors.get('reference', '#ff7f0e'),
                   linestyle='--',
                   alpha=line_style.get('reference_alpha', 0.5),
                   label='Reference')
        
        # Add labels and title
        ax.set_xlabel(figure_data.get('xlabel', 'X'))
        ax.set_ylabel(figure_data.get('ylabel', 'Y'))
        if group_label:
            ax.set_title(f"{group_label} (n={figure_data.get('n_obs', '')})")
        
        # Add metric value if present
        if 'value' in figure_data:
            ax.text(0.95, 0.05,
                   f"Value: {figure_data['value']:.3f}",
                   transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='bottom')
        
        ax.legend()

    def _calculate_distribution(self, figure_data: Dict, style: Optional[Dict] = None) -> Dict:
        """
        Plot or calculate the distribution/association visualization with consistent styling.
        
        Parameters
        ----------
        figure_data : Dict
            Dictionary containing data for visualization
        style : Dict, optional
            Style configuration dictionary; if no style is provided, default template is used
            
        Returns
        -------
        Dict
            Dictionary containing base64 encoded image and dimensions
        """
        if style is None:
            style = self._style
        
        # Extract style configurations
        colors = style.get('colors', {})
        fig_style = style.get('figure', {})
        grid_style = style.get('grid', {})
        line_style = style.get('lines', {})
        hist_style = style.get('histogram', {})
        legend_style = style.get('legend', {})
        
        # Create figure with consistent size
        fig_size = fig_style.get('size', [8, 5])
        fig, ax = plt.subplots(figsize=fig_size)
        
        # if self.__class__.__name__ == "ScoreHistogram":
        #     bin_edges = figure_data['bin_edges']
        #     hist_def = figure_data['hist_defaulted']
        #     hist_nondef = figure_data['hist_non_defaulted']
            
        #     # Plot histograms with consistent colors and style
        #     ax.hist(bin_edges[:-1], bins=bin_edges, weights=hist_nondef, 
        #            alpha=hist_style.get('alpha', 0.5),
        #            color=colors.get('non_defaulted', '#2ca02c'),
        #            label='Non-defaulted')
        #     ax.hist(bin_edges[:-1], bins=bin_edges, weights=hist_def,
        #            alpha=hist_style.get('alpha', 0.5),
        #            color=colors.get('defaulted', '#d62728'),
        #            label='Defaulted')
                   
        #     ax.set_xlabel("Score", fontsize=fig_style.get('label_fontsize', 10))
        #     ax.set_ylabel("Count", fontsize=fig_style.get('label_fontsize', 10))
        #     ax.set_title("PD Model Scores Distribution", fontsize=fig_style.get('title_fontsize', 12))
            
        # elif self.__class__.__name__ == "KSDistPlot":
        #     x = figure_data['thresholds']
        #     cdf_good = np.asarray(figure_data['cdf_non_defaulted'])
        #     cdf_bad = np.asarray(figure_data['cdf_defaulted'])
            
        #     # Plot CDFs with consistent style
        #     ax.plot(x, cdf_good, 
        #            color=colors.get('non_defaulted', '#2ca02c'),
        #            linewidth=line_style.get('main_width', 2),
        #            alpha=line_style.get('main_alpha', 0.8),
        #            label='Non-defaulted CDF')
        #     ax.plot(x, cdf_bad,
        #            color=colors.get('defaulted', '#d62728'),
        #            linewidth=line_style.get('main_width', 2),
        #            alpha=line_style.get('main_alpha', 0.8),
        #            label='Defaulted CDF')
                   
        #     if 'ks_stat' in figure_data and 'ks_threshold' in figure_data:
        #         ks_stat = figure_data['ks_stat']
        #         ks_x = figure_data['ks_threshold']
        #         ax.vlines(ks_x, 
        #                  cdf_good[np.argmax(np.abs(cdf_good - cdf_bad))],
        #                  cdf_bad[np.argmax(np.abs(cdf_good - cdf_bad))],
        #                  color=colors.get('reference_line', '#ff7f0e'),
        #                  linestyle='--',
        #                  alpha=line_style.get('reference_alpha', 0.5),
        #                  label=f'KS={ks_stat:.3f}')
                         
        #     ax.set_xlabel("Score", fontsize=fig_style.get('label_fontsize', 10))
        #     ax.set_ylabel("Cumulative Proportion", fontsize=fig_style.get('label_fontsize', 10))
        #     ax.set_title("KS Distribution Plot", fontsize=fig_style.get('title_fontsize', 12))
            
        # elif self.__class__.__name__ == "PDLiftPlot":
        #     percentiles = figure_data['percentiles']
        #     lift = figure_data['lift']
            
        #     # Plot lift curve with consistent style
        #     ax.plot(percentiles, lift,
        #            color=colors.get('palette', ['#1f77b4'])[0],
        #            alpha=line_style.get('main_alpha', 0.8),
        #            linewidth=line_style.get('main_width', 2),
        #            label='Lift')
            
        #     # Add reference line
        #     ax.axhline(y=1,
        #               color=colors.get('reference_line', '#ff7f0e'),
        #               linestyle='--',
        #               alpha=line_style.get('reference_alpha', 0.5),
        #               linewidth=line_style.get('reference_width', 1.5),
        #               label='Random Model')
            
        #     ax.set_xlabel("Population Percentile", fontsize=fig_style.get('label_fontsize', 10))
        #     ax.set_ylabel("Lift (Relative Default Rate)", fontsize=fig_style.get('label_fontsize', 10))
        #     ax.set_title("Default Rate Lift", fontsize=fig_style.get('title_fontsize', 12))
            
        #     # Set percentage ticks
        #     step = style.get('axes', {}).get('percentages', {}).get('step', 20)
        #     ax.set_xticks(np.arange(0, 101, step))
        #     ax.set_xticklabels([f"{x}%" for x in range(0, 101, step)])
            
        # elif self.__class__.__name__ == "PDGainPlot":
        #     percentiles = figure_data['percentiles']
        #     gains = figure_data['gains']
            
        #     # Plot gain curve with consistent style
        #     ax.plot(percentiles, gains,
        #            color=colors.get('palette', ['#1f77b4'])[0],
        #            alpha=line_style.get('main_alpha', 0.8),
        #            linewidth=line_style.get('main_width', 2),
        #            label='Actual')
            
        #     # Add reference line
        #     if style.get('axes', {}).get('show_diagonal', True):
        #         ax.plot([0, 100], [0, 100],
        #                color=colors.get('reference_line', '#ff7f0e'),
        #                linestyle='--',
        #                alpha=line_style.get('reference_alpha', 0.5),
        #                linewidth=line_style.get('reference_width', 1.5),
        #                label='Random')
            
        #     ax.set_xlabel("Population Percentile", fontsize=fig_style.get('label_fontsize', 10))
        #     ax.set_ylabel("Cumulative % of Defaults Captured", fontsize=fig_style.get('label_fontsize', 10))
        #     ax.set_title("Gain Chart", fontsize=fig_style.get('title_fontsize', 12))
            
        #     # Set percentage ticks
        #     step = style.get('axes', {}).get('percentages', {}).get('step', 20)
        #     ax.set_xlim(0, 100)
        #     ax.set_ylim(0, 100)
        #     ax.set_xticks(np.arange(0, 101, step))
        #     ax.set_yticks(np.arange(0, 101, step))
        #     ax.set_xticklabels([f"{x}%" for x in range(0, 101, step)])
        #     ax.set_yticklabels([f"{y}%" for y in range(0, 101, step)])
        #     ax.legend(loc='lower right')
        # else:
        #     raise NotImplementedError(f"Distribution plotting for {self.__class__.__name__} is not available.")
            
        # # Apply common styling to all plots
        # if grid_style.get('show', True):
        #     ax.grid(True,
        #            linestyle=grid_style.get('linestyle', '--'),
        #            alpha=grid_style.get('alpha', 0.3),
        #            color=grid_style.get('color', '#cccccc'))
                   
        # # Configure legend with consistent style
        # ax.legend(fontsize=legend_style.get('fontsize', 10),
        #          framealpha=legend_style.get('framealpha', 0.8))
                 
        # # Set tick label sizes
        # ax.tick_params(labelsize=fig_style.get('tick_fontsize', 8))
        self._plot_figure(ax, figure_data, style)
        # Convert matplotlib figure to a dictionary representation
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

    def show_plot(self, figure_data: Dict, style: Optional[Dict]=None):
        """
        Display the plot for the distribution/association metric.
        """
        plot_data = self._calculate_distribution(figure_data, style)
        img_data = base64.b64decode(plot_data["image_base64"])
        img = plt.imread(BytesIO(img_data))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def save_plot(self, figure_data: Dict, filepath: str, style: Optional[Dict]=None):
        """
        Save the plot to disk.
        
        Parameters
        ----------
        figure_data : Dict
            The data to plot
        filepath : str
            Path where to save the plot
        style : Dict, optional
            Style configuration to use for the plot
        """
        plot_data = self._calculate_distribution(figure_data, style)
        img_data = base64.b64decode(plot_data["image_base64"])
        with open(filepath, "wb") as f:
            f.write(img_data)

