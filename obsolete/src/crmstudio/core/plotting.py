"""
Centralized plotting service for CRMStudio.

This module provides a unified plotting mechanism for all metrics,
reducing code duplication and providing consistent visualization styling.
"""

import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from typing import Dict, Optional, Any, List, Tuple, Union, Generator
import pandas as pd
from contextlib import contextmanager

from .data_classes import MetricResult

class PlottingService:
    """
    Centralized service for all plotting operations in CRMStudio.
    
    This class handles visualization needs for metrics including:
    - Curve plots (ROC, CAP)
    - Distribution plots (histograms, KS plot)
    - Group analysis plots (time series, segments)
    """
    
    def __init__(self, style: Optional[Dict] = None):
        """Initialize the plotting service with a style configuration."""
        self.style = style or self._load_default_style()
    
    def _load_default_style(self) -> Dict:
        """Load default plotting style from configuration."""
        from .config_loader import load_config
        default_style_path = "src/crmstudio/core/templates/figure_style.yaml"
        return load_config(default_style_path)
    
    def set_style(self, style: Dict):
        """Update the plotting style."""
        self.style = style

    @contextmanager
    def _figure_context(self, *args, **kwargs) -> Generator[Tuple[plt.Figure, plt.Axes], None, None]:
        """
        Context manager for creating and properly cleaning up matplotlib figures.
        
        This ensures that figures are properly closed even if exceptions occur
        during plotting operations, preventing memory leaks and resource issues.
        
        Parameters
        ----------
        *args, **kwargs
            Arguments passed to plt.subplots()
            
        Yields
        ------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects for plotting
        """
        fig, ax = plt.subplots(*args, **kwargs)
        try:
            yield fig, ax
        finally:
            plt.close(fig)
    
    @contextmanager
    def _multi_figure_context(self, rows: int, cols: int, figsize=None, **kwargs) -> Generator[Tuple[plt.Figure, np.ndarray], None, None]:
        """
        Context manager for creating and properly cleaning up matplotlib figures with multiple subplots.
        
        This is similar to _figure_context but specialized for multiple subplots.
        
        Parameters
        ----------
        rows : int
            Number of rows in the subplot grid
        cols : int
            Number of columns in the subplot grid
        figsize : tuple, optional
            Figure size in inches (width, height)
        **kwargs
            Additional arguments passed to plt.subplots()
            
        Yields
        ------
        Tuple[plt.Figure, np.ndarray]
            Figure and array of axes objects for plotting
        """
        fig, axes = plt.subplots(rows, cols, figsize=figsize, **kwargs)
        try:
            yield fig, axes
        finally:
            plt.close(fig)
    
    def _apply_common_styling(self, ax):
        """Apply common styling elements to an axis."""
        fig_style = self.style.get('figure', {})
        grid_style = self.style.get('grid', {})
        
        # Apply grid if specified
        if grid_style.get('show', True):
            ax.grid(
                True,
                linestyle=grid_style.get('linestyle', '--'),
                alpha=grid_style.get('alpha', 0.3),
                color=grid_style.get('color', '#cccccc')
            )
        
        # Set tick label sizes
        ax.tick_params(labelsize=fig_style.get('tick_fontsize', 8))

    
    def _convert_to_image(self, fig: plt.Figure) -> Dict[str, Any]:
        """Convert a matplotlib figure to a dictionary with image data."""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        # No need to close the figure here as it's handled by the context manager
        
        return {
            "image_base64": image_base64,
            "width": fig.get_size_inches()[0] * fig.dpi,
            "height": fig.get_size_inches()[1] * fig.dpi
        }
    
    @contextmanager
    def _plotting_context(self):
        """
        Context manager for the entire plotting process.
        
        This handles any global setup/teardown needed for plotting operations
        and ensures cleanup even if exceptions occur during the plotting process.
        
        Yields
        ------
        None
            This context manager doesn't yield any value
        """
        # Save the original matplotlib settings
        original_style = plt.rcParams.copy()
        
        try:
            # Apply any global styles or settings here if needed
            yield
        finally:
            # Reset to original matplotlib settings
            plt.rcParams.update(original_style)
            # Ensure all figures are closed (safety measure)
            plt.close('all')
    
    def plot(self, metric_result: MetricResult, plot_type: str = None) -> Dict[str, Any]:
        """
        Plot a figure based on metric_result and the specified plot type.
        
        This is the main entry point for all plotting in the PlottingService.
        It handles all metrics, always treating results as a collection of data series
        where a single curve is just a special case with one item.
        
        Parameters
        ----------
        metric_result : MetricResult
            MetricResult object containing data for visualization
        plot_type : str, optional
            The type of plot ('curve', 'distribution', 'calibration', 'psi', 'csi',
            'drift', 'migration', 'rating_stability'). If None, will be detected from data.
            
        Returns
        -------
        Dict
            Dictionary with image data
        """
        if metric_result is None:
            raise ValueError("No metric result provided for plotting.")
        
        with self._plotting_context():
            if metric_result.has_figure():
                # Get standardized datasets directly from MetricResult
                datasets = metric_result.prepare_for_plotting()
                
                # Determine plot type based on data if not specified
                if plot_type is None:
                    if metric_result.is_scalar_collection():
                        plot_type = 'scalar'
                    elif metric_result.is_curve_data():
                        plot_type = 'curve'
                    elif 'plot_type' in datasets[0]:
                        # Get plot_type from the first dataset if available
                        first_data = next(iter(datasets[0].values()))
                        plot_type = first_data.get('plot_type', 'distribution')
                    else:
                        plot_type = 'distribution'
                
                # Route to appropriate plotting function based on plot_type
                if plot_type == 'scalar' or metric_result.is_scalar_collection():
                    fig, ax = self._plot_scalar_collection(datasets)
                elif plot_type == 'curve':
                    # attempt plotting multiple curves at single chart
                    fig, ax = self._plot_curve(datasets)
                elif plot_type == 'calibration':
                    # Handle calibration plots
                    fig, ax = self._plot_calibration(datasets)
                elif plot_type == 'distribution':
                    # create as many subplots as there are elements in datasets list
                    fig, ax = self._plot_distribution(datasets)
                elif plot_type == 'psi':
                    # Population Stability Index plot
                    fig, ax = self._plot_psi(datasets)
                elif plot_type == 'csi':
                    # Characteristic Stability Index plot
                    fig, ax = self._plot_csi(datasets)
                elif plot_type == 'drift':
                    # Temporal Drift Detection plot
                    fig, ax = self._plot_drift(datasets)
                elif plot_type == 'migration':
                    # Migration Analysis plot
                    fig, ax = self._plot_migration(datasets)
                elif plot_type == 'rating_stability':
                    # Rating Stability Analysis plot
                    fig, ax = self._plot_rating_stability(datasets)
                elif plot_type == 'concentration':
                    # Concentration (Herfindahl Index) plot
                    fig, ax = self._plot_concentration(datasets)
                else:
                    raise ValueError(f"Unknown plot type: {plot_type}")
            else:
                # scalar method was called, we will plot a bar chart with one bar only
                fig, ax = self._plot_scalar_collection(metric_result)
    
            self._apply_common_styling(ax)
            image_data = self._convert_to_image(fig)
    
            # Return the image data
            return image_data

    def _plot_scalar_collection(self, results: Union[MetricResult, List[Dict[str, Any]]]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a collection of scalar metrics as a bar chart."""
        with self._figure_context() as (fig, ax):
            if isinstance(results, MetricResult):
                # Get standardized datasets directly from MetricResult
                datasets = results.prepare_for_plotting()
                labels = []
                values = []
                
                for dataset in datasets:
                    for label, value in dataset.items():
                        labels.append(label)
                        values.append(value if not isinstance(value, dict) else value.get('value', 0))
                
                # Create the bar chart
                ax.bar(labels, values)
            else:
                # List of dictionaries (already prepared datasets)
                labels = []
                values = []
                
                for dataset in results:
                    for label, value in dataset.items():
                        # Convert any datetime/period objects to strings to avoid plotting issues
                        if isinstance(label, (pd.Period, pd.Timestamp)):
                            label = str(label)
                        
                        labels.append(label)
                        values.append(value if not isinstance(value, dict) else value.get('value', 0))
                
                # Create the bar chart
                ax.bar(labels, values)
            
            # Adjust layout for readability if many labels
            if len(labels) > 3:
                plt.xticks(rotation=45, ha='right')
                fig.tight_layout()
                    
        return fig, ax

    def _plot_curve(self, datasets: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a curve based on the provided datasets."""
        with self._figure_context() as (fig, ax):
            # Plot the diagonal reference line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            
            # Plot each dataset
            for dataset in datasets:
                for label, data in dataset.items():
                    ax.plot(data['x'], data['y'], label=label)
            
            ax.legend()
        return fig, ax

    def _plot_distribution(self, datasets: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a distribution based on the provided datasets.
        Here we have following options:
        - Histogram for ScoreHistogram
        - 2 curves for KSDistPlot
        - 1 curve for PDGain and PDLift
        """
        # Number of subplots:
        n_subplots = len(datasets)
        
        with self._multi_figure_context(n_subplots, 1, figsize=(8, 4 * n_subplots)) as (fig, axs):
            if n_subplots == 1:
                axs = [axs]  # Ensure axs is always a list
    
            for ax, dataset in zip(axs, datasets):
                for label, data in dataset.items():
                    ax.set_title(label)
                    # Only histogram data contain x_def and y_def
                    if 'x_def' in data and 'y_def' in data:
                        ax.hist(x = data['x_def'], bins = data['bin_edges'], weights=data['y_def'], label="Defaulted", alpha=0.5)
                        ax.hist(x = data['x_ndef'], bins = data['bin_edges'], weights=data['y_ndef'], label="Non-defaulted", alpha=0.5)
                    # All other metrics return x, y, x_ref and y_ref, for them we plot simple curve 
                    else:
                        if 'actual_label' in data:
                            main_label = data['actual_label']
                            ref_label = data.get('ref_label', 'CDF nondefaulted')
                        else:
                            main_label = "Model"
                            ref_label = "Random"
                        ax.plot(data['x'], data['y'], label=main_label)
                        if 'x_ref' in data and 'y_ref' in data:
                            ax.plot(data['x_ref'], data['y_ref'], label=ref_label, linestyle='--')
                    ax.legend()
        return fig, axs[0]
        
    def _plot_calibration(self, datasets: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot calibration metrics (reliability diagrams, Hosmer-Lemeshow plots).
        
        This specialized plotting function handles the unique needs of calibration plots
        including reference lines, annotations, and different plot styles.
        
        Parameters
        ----------
        datasets : List[Dict[str, Any]]
            List of datasets to plot, each containing calibration data
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects with the rendered plot
        """
        # Number of subplots - we'll plot one per dataset
        n_subplots = len(datasets)
        
        with self._multi_figure_context(n_subplots, 1, figsize=(10, 5 * n_subplots)) as (fig, axs):
            if n_subplots == 1:
                axs = [axs]  # Ensure axs is always a list
                
            for ax, dataset in zip(axs, datasets):
                for label, data in dataset.items():
                    # Set the title if provided, otherwise use the dataset label
                    title = data.get('title', label)
                    ax.set_title(title, fontsize=12)
                    
                    # Add axis labels if provided
                    if 'xlabel' in data:
                        ax.set_xlabel(data['xlabel'])
                    if 'ylabel' in data:
                        ax.set_ylabel(data['ylabel'])
                    
                    # Check plot type - bubble plot for ECE
                    if data.get('plot_type') == 'bubble':
                        # Plot perfect calibration reference line
                        ax.plot(
                            data['x_ref'], 
                            data['y_ref'], 
                            'k--', 
                            alpha=0.5, 
                            label=data.get('ref_label', 'Perfect Calibration')
                        )
                        
                        # Create bubble plot where size represents weight
                        weights = data.get('weights', [1.0] * len(data['x']))
                        size_scale = 1000  # Scale factor for bubble size
                        scatter = ax.scatter(
                            data['x'], 
                            data['y'],
                            s=[w * size_scale for w in weights],
                            alpha=0.6,
                            label=data.get('actual_label', 'Observed'),
                            zorder=3
                        )
                        
                        # Add rating names as annotations if provided
                        if 'rating_names' in data:
                            for i, txt in enumerate(data['rating_names']):
                                ax.annotate(
                                    txt, 
                                    (data['x'][i], data['y'][i]),
                                    xytext=(5, 0),
                                    textcoords='offset points',
                                    fontsize=8
                                )
                        
                        # # Add size legend if needed
                        # if max(weights) - min(weights) > 0.05:  # Only if weights vary enough
                        #     handles, labels = scatter.legend_elements(
                        #         prop="sizes", 
                        #         alpha=0.6,
                        #         num=3,
                        #         func=lambda s: s/size_scale
                        #     )
                        #     ax.legend(handles, labels, loc="upper left", title="Sample Weight")
                    
                    # Jeffreys test plot - confidence intervals around observed default rates
                    elif data.get('plot_type') == 'jeffreys':
                        # Get data from the dataset
                        x = data['x']  # Rating positions or single portfolio point
                        y = data['y']  # Observed default rates
                        y_pred = data['y_pred']  # Predicted PDs
                        y_lower = data['y_lower']  # Lower confidence bounds
                        y_upper = data['y_upper']  # Upper confidence bounds
                        
                        # Create the confidence interval plot
                        if len(x) == 1:  # Single portfolio point (no ratings)
                            # Plot confidence interval as a bar
                            ax.bar(x, y, label='Observed DR', alpha=0.6, width=0.3)
                            ax.scatter(x, y_pred, color='red', marker='x', s=100, label='Predicted PD')
                            ax.vlines(x, y_lower, y_upper, color='black', linewidth=1.5, label='95% CI')
                            
                            # Add horizontal line at the bounds
                            for i in range(len(x)):
                                ax.hlines(y_lower[i], x[i]-0.2, x[i]+0.2, color='black', linewidth=1.5)
                                ax.hlines(y_upper[i], x[i]-0.2, x[i]+0.2, color='black', linewidth=1.5)
                                
                            # Set labels
                            ax.set_xticks([])  # No x-axis ticks for single point
                        else:
                            # Plot for multiple ratings
                            ax.bar(x, y, label='Observed DR', alpha=0.6, width=0.7)
                            ax.scatter(x, y_pred, color='red', marker='x', s=100, label='Predicted PD')
                            
                            # Plot confidence intervals
                            for i in range(len(x)):
                                ax.vlines(x[i], y_lower[i], y_upper[i], color='black', linewidth=1.5)
                                ax.hlines(y_lower[i], x[i]-0.3, x[i]+0.3, color='black', linewidth=1.5)
                                ax.hlines(y_upper[i], x[i]-0.3, x[i]+0.3, color='black', linewidth=1.5)
                            
                            # Set x-axis labels if provided
                            if 'rating_labels' in data:
                                ax.set_xticks(x)
                                ax.set_xticklabels(data['rating_labels'], rotation=45, ha='right')
                        
                        # Add a legend
                        ax.legend(loc='best')
                        
                        # Set y-axis as percentage
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
                        
                        # Adjust y limits to include confidence intervals with some padding
                        if y_lower and y_upper:
                            y_min = min(y_lower) * 0.9
                            y_max = max(y_upper) * 1.1
                            ax.set_ylim([max(0, y_min), min(1, y_max)])
                    
                    # Binomial test plot - observed vs expected defaults
                    elif data.get('plot_type') == 'binomial':
                        # Get data from the dataset
                        x = data['x']  # Rating positions or single portfolio point
                        y = data['y']  # Observed defaults
                        y_expected = data['y_expected']  # Expected defaults
                        
                        # Create the bar chart comparison
                        bar_width = 0.35
                        
                        if len(x) == 1:  # Single portfolio point (no ratings)
                            # Plot two bars side by side for observed and expected
                            ax.bar(x[0] - bar_width/2, y[0], bar_width, label='Observed Defaults', color='blue', alpha=0.7)
                            ax.bar(x[0] + bar_width/2, y_expected[0], bar_width, label='Expected Defaults', color='red', alpha=0.7)
                            
                            # Set labels
                            ax.set_xticks([x[0]])
                            ax.set_xticklabels(['Portfolio'])
                        else:
                            # Plot for multiple ratings
                            x_pos = np.array(x)
                            ax.bar(x_pos - bar_width/2, y, bar_width, label='Observed Defaults', color='blue', alpha=0.7)
                            ax.bar(x_pos + bar_width/2, y_expected, bar_width, label='Expected Defaults', color='red', alpha=0.7)
                            
                            # Set x-axis labels if provided
                            if 'rating_labels' in data:
                                ax.set_xticks(x)
                                ax.set_xticklabels(data['rating_labels'], rotation=45, ha='right')
                        
                        # Add a legend
                        ax.legend(loc='best')
                        
                        # Adjust y limits to include all bars with some padding
                        y_max = max(max(y), max(y_expected)) * 1.1
                        ax.set_ylim([0, y_max])
                        
                        # Add count labels on top of bars
                        for i in range(len(x)):
                            ax.text(x[i] - bar_width/2, y[i] + 0.05*y_max, f'{int(y[i])}', 
                                   ha='center', va='bottom', fontsize=9)
                            ax.text(x[i] + bar_width/2, y_expected[i] + 0.05*y_max, f'{y_expected[i]:.1f}', 
                                   ha='center', va='bottom', fontsize=9)
                    
                    # Rating Heterogeneity Test plot - default rates across rating grades with significance markers
                    elif data.get('plot_type') == 'rating_heterogeneity':
                        # Get data from the dataset
                        x = data['x']  # Rating grades
                        y = data['y']  # Default rates
                        n_obs = data.get('n_obs', [100] * len(x))  # Sample sizes for marker scaling
                        significant_markers = data.get('significant_markers', [None] * len(x))  # Significance markers
                        
                        # Plot default rates as a line with markers
                        line = ax.plot(x, y, 'o-', markersize=8, label='Default Rate', zorder=3)
                        line_color = line[0].get_color()
                        
                        # Adjust y-axis to show percentages
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
                        
                        # Add significance markers between ratings
                        for i in range(len(x) - 1):
                            if significant_markers[i] is not None:
                                x_mid = (i + i + 1) / 2
                                y_mid = (y[i] + y[i+1]) / 2
                                color = 'green' if significant_markers[i] else 'red'
                                marker = '✓' if significant_markers[i] else '✗'
                                ax.plot([i, i+1], [y[i], y[i+1]], '-', color=color, alpha=0.5, linewidth=2)
                                ax.text(x_mid, y_mid, marker, 
                                       ha='center', va='center', fontsize=12, color=color,
                                       bbox=dict(boxstyle="circle", fc="white", ec=color, alpha=0.8))
                        
                        # Set x-axis labels
                        ax.set_xticks(range(len(x)))
                        ax.set_xticklabels(x, rotation=45, ha='right')
                        
                        # Add second y-axis for p-values if provided
                        if 'p_values' in data:
                            ax2 = ax.twinx()
                            ax2.plot(range(len(x) - 1), data['p_values'], 'r--', alpha=0.6, label='p-value')
                            ax2.set_ylabel('p-value')
                            ax2.grid(False)
                        
                        # Add a legend
                        ax.legend(loc='upper left')
                        
                        # Set y limits to include all data with padding
                        y_max = max(y) * 1.2
                        ax.set_ylim([0, min(1, y_max)])
                    
                    # Rating Homogeneity Test plot - p-values and relative ranges by rating grade
                    elif data.get('plot_type') == 'rating_homogeneity':
                        # Get data from the dataset
                        x = data['x']  # Rating grades
                        y = data['y']  # p-values
                        relative_ranges = data.get('relative_ranges', [0] * len(x))  # Relative ranges
                        passed = data.get('passed', [True] * len(x))  # Test results
                        
                        # Plot p-values as bars with color indicating test result
                        colors = ['green' if p else 'red' for p in passed]
                        ax.bar(range(len(x)), y, color=colors, alpha=0.7, label='p-value')
                        
                        # Add horizontal line at significance level (typically 0.05)
                        alpha = 0.05  # Assuming standard 95% confidence level
                        ax.axhline(y=alpha, color='red', linestyle='--', alpha=0.8, label=f'α = {alpha}')
                        
                        # Set x-axis labels
                        ax.set_xticks(range(len(x)))
                        ax.set_xticklabels(x, rotation=45, ha='right')
                        
                        # Add second y-axis for relative ranges if provided
                        if len(relative_ranges) == len(x):
                            ax2 = ax.twinx()
                            ax2.plot(range(len(x)), relative_ranges, 'bo-', alpha=0.6, label='Relative Range')
                            ax2.set_ylabel('Relative Range of Default Rates')
                            ax2.grid(False)
                            
                            # Add legend for second y-axis
                            lines, labels = ax2.get_legend_handles_labels()
                            ax.legend(loc='upper left')
                            ax2.legend(lines, labels, loc='upper right')
                        else:
                            ax.legend(loc='best')
                        
                        # Plot sub-bucket data if available
                        if 'buckets_data' in data:
                            # Check if we need a second figure for detailed bucket view
                            buckets_data = data['buckets_data']
                            if buckets_data and len(buckets_data) > 0:
                                # We could create a secondary visualization here for bucket details
                                # But for now, we'll just add annotations on the main plot
                                pass
                    
                    else:
                        # Standard curve plot for Hosmer-Lemeshow and calibration curves
                        ax.plot(
                            data['x'], 
                            data['y'], 
                            'o-', 
                            label=data.get('actual_label', 'Observed')
                        )
                        
                        # Plot reference curve if available
                        if 'x_ref' in data and 'y_ref' in data:
                            ax.plot(
                                data['x_ref'], 
                                data['y_ref'], 
                                'k--', 
                                alpha=0.5, 
                                label=data.get('ref_label', 'Reference')
                            )
                    
                    # Add annotations if provided
                    if 'annotations' in data:
                        y_pos = 0.02
                        for annotation in data['annotations']:
                            ax.annotate(
                                annotation, 
                                xy=(0.02, y_pos), 
                                xycoords='axes fraction',
                                fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                            )
                            y_pos += 0.06  # Move up for next annotation
                    
                    # Set axis limits if provided
                    if 'xlim' in data:
                        ax.set_xlim(data['xlim'])
                    if 'ylim' in data:
                        ax.set_ylim(data['ylim'])
                    else:
                        # For calibration plots, often want 0-1 range on both axes
                        if data.get('plot_type') not in ['hosmer_lemeshow', 'rating_heterogeneity', 'rating_homogeneity', 'binomial']:
                            ax.set_xlim([0, 1])
                            ax.set_ylim([0, 1])
                    
                    # Use percentage ticks if specified
                    if data.get('use_percentage_ticks', False):
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                    if data.get('use_percentage_ticks_y', False):
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                    
                    # Add legend if not already added for special cases
                    if data.get('plot_type') not in ['rating_homogeneity']:
                        ax.legend(loc='best')
                        
                    ax.grid(True, linestyle='--', alpha=0.3)
        
        return fig, axs[0]

    def display_image(self, image_data: Dict):
        """Display an image in the current context."""
        img_data = base64.b64decode(image_data["image_base64"])
        
        with self._figure_context(figsize=(image_data.get("width", 800)/100, 
                                          image_data.get("height", 600)/100)) as (fig, ax):
            img = plt.imread(BytesIO(img_data))
            ax.imshow(img)
            ax.axis('off')
            plt.tight_layout()
            plt.show()
    
    def save_image(self, image_data: Dict, filepath: str):
        """Save an image to disk."""
        img_data = base64.b64decode(image_data["image_base64"])
        with open(filepath, "wb") as f:
            f.write(img_data)
            
    def _plot_psi(self, datasets: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Population Stability Index (PSI) visualization.
        
        Parameters
        ----------
        datasets : List[Dict[str, Any]]
            List of datasets to plot, each containing PSI data
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects with the rendered plot
        """
        with self._figure_context(figsize=(10, 6)) as (fig, ax):
            # We expect a single dataset for PSI
            dataset = datasets[0]
            data = next(iter(dataset.values()))
            
            # Get the bin information
            x = data.get('x', [])  # Bin numbers
            y_ref = data.get('y_ref', [])  # Reference proportions
            y_recent = data.get('y_recent', [])  # Recent proportions
            bin_psi = data.get('bin_psi', [])  # PSI contributions by bin
            bin_edges = data.get('bin_edges', [])  # Bin boundaries
            
            # Plot proportions as bars
            bar_width = 0.35
            ax.bar(np.array(x) - bar_width/2, y_ref, bar_width, label='Reference', alpha=0.7, color='blue')
            ax.bar(np.array(x) + bar_width/2, y_recent, bar_width, label='Recent', alpha=0.7, color='orange')
            
            # Add a second axis for PSI contributions
            ax2 = ax.twinx()
            
            # Plot PSI contributions as a line
            ax2.plot(x, bin_psi, 'r-o', label='PSI Contribution')
            
            # Set axis labels and title
            ax.set_xlabel(data.get('xlabel', 'Score/PD Bins'))
            ax.set_ylabel(data.get('ylabel', 'Proportion of Population'))
            ax2.set_ylabel('PSI Contribution')
            ax.set_title(data.get('title', 'Population Stability Index (PSI)'))
            
            # Set x-ticks to bin boundaries if available
            if bin_edges and len(bin_edges) > 1:
                # Calculate midpoints for tick labels
                midpoints = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
                ax.set_xticks(x)
                ax.set_xticklabels([f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)], 
                                  rotation=45, ha='right')
            
            # Add annotations if provided
            if 'annotations' in data:
                y_pos = 0.02
                for annotation in data['annotations']:
                    ax.annotate(
                        annotation, 
                        xy=(0.02, y_pos), 
                        xycoords='axes fraction',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
                    y_pos += 0.06  # Move up for next annotation
            
            # Add legends for both axes
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            # Set grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
        return fig, ax
    
    def _plot_csi(self, datasets: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Characteristic Stability Index (CSI) visualization.
        
        Parameters
        ----------
        datasets : List[Dict[str, Any]]
            List of datasets to plot, each containing CSI data
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects with the rendered plot
        """
        with self._figure_context(figsize=(10, 6)) as (fig, ax):
            # We expect a single dataset for CSI
            dataset = datasets[0]
            data = next(iter(dataset.values()))
            
            # Get the variable data
            x = data.get('x', [])  # Variable names
            y = data.get('y', [])  # CSI values
            threshold = data.get('threshold', 0.1)  # CSI threshold
            
            # Create horizontal bar chart for CSI values
            bars = ax.barh(x, y, alpha=0.7)
            
            # Color bars based on threshold
            for i, bar in enumerate(bars):
                if y[i] >= 0.2:  # Red for significant shift
                    bar.set_color('red')
                elif y[i] >= 0.1:  # Yellow/orange for moderate shift
                    bar.set_color('orange')
                else:  # Green for stable
                    bar.set_color('green')
            
            # Add threshold line
            ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, 
                      label=f'Threshold ({threshold})')
            
            # Add a more strict threshold line at 0.2
            ax.axvline(x=0.2, color='darkred', linestyle=':', alpha=0.7, 
                      label='Critical (0.2)')
            
            # Set axis labels and title
            ax.set_xlabel('CSI Value')
            ax.set_ylabel('Variables')
            ax.set_title(data.get('title', 'Characteristic Stability Index (CSI)'))
            
            # Add value labels to the bars
            for i, v in enumerate(y):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
            
            # Add annotations if provided
            if 'annotations' in data:
                y_pos = 0.02
                for annotation in data['annotations']:
                    ax.annotate(
                        annotation, 
                        xy=(0.02, y_pos), 
                        xycoords='axes fraction',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
                    y_pos += 0.06  # Move up for next annotation
            
            # Add legend
            ax.legend()
            
            # Set grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Adjust layout
            fig.tight_layout()
            
        return fig, ax
    
    def _plot_drift(self, datasets: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Temporal Drift Detection visualization.
        
        Parameters
        ----------
        datasets : List[Dict[str, Any]]
            List of datasets to plot, each containing drift data
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects with the rendered plot
        """
        with self._figure_context(figsize=(10, 6)) as (fig, ax):
            # We expect a single dataset for drift detection
            dataset = datasets[0]
            data = next(iter(dataset.values()))
            
            # Get the time series data
            x = data.get('x', [])  # Time points
            y = data.get('y', [])  # Metric values
            trend_line = data.get('trend_line', [])  # Trend line
            future_point = data.get('future_point', [])  # Future prediction
            
            # Plot the actual metric values over time
            ax.plot(x, y, 'o-', label=data.get('ylabel', 'Performance Metric'), linewidth=2)
            
            # Plot trend line if available
            if trend_line and len(trend_line) == len(x):
                ax.plot(x, trend_line, 'r--', label='Trend', alpha=0.7)
            
            # Plot future prediction point if available
            if future_point and len(future_point) == 2:
                next_x = future_point[0]
                next_y = future_point[1]
                ax.plot([next_x], [next_y], 'rx', markersize=10, label='Next Period Prediction')
                
                # Connect to the last actual point with a dotted line
                if len(x) > 0 and len(y) > 0:
                    ax.plot([x[-1], next_x], [y[-1], next_y], 'r:', alpha=0.5)
            
            # Set axis labels and title
            ax.set_xlabel(data.get('xlabel', 'Time'))
            ax.set_ylabel(data.get('ylabel', 'Performance Metric'))
            ax.set_title(data.get('title', 'Temporal Drift Analysis'))
            
            # Format x-axis ticks for time series
            if len(x) > 0 and isinstance(x[0], (str, pd.Timestamp, pd.Period)):
                plt.xticks(rotation=45, ha='right')
            
            # Add annotations if provided
            if 'annotations' in data:
                y_pos = 0.02
                for annotation in data['annotations']:
                    ax.annotate(
                        annotation, 
                        xy=(0.02, y_pos), 
                        xycoords='axes fraction',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
                    y_pos += 0.06  # Move up for next annotation
            
            # Add legend
            ax.legend()
            
            # Set grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Adjust layout
            fig.tight_layout()
            
        return fig, ax
    
    def _plot_migration(self, datasets: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Rating Migration Matrix visualization.
        
        Parameters
        ----------
        datasets : List[Dict[str, Any]]
            List of datasets to plot, each containing migration data
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects with the rendered plot
        """
        with self._figure_context(figsize=(10, 8)) as (fig, ax):
            # We expect a single dataset for migration analysis
            dataset = datasets[0]
            data = next(iter(dataset.values()))
            
            # Get the migration matrix data
            matrix = data.get('matrix', [])  # Migration probability matrix
            rating_scale = data.get('rating_scale', [])  # Rating scale
            
            # Convert to numpy array if it's not already
            matrix = np.array(matrix)
            
            # Create heatmap
            im = ax.imshow(matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Transition Probability')
            
            # Set axis labels and title
            ax.set_xlabel(data.get('xlabel', 'Current Rating'))
            ax.set_ylabel(data.get('ylabel', 'Initial Rating'))
            ax.set_title(data.get('title', 'Rating Migration Matrix'))
            
            # Set tick labels to rating scale if provided
            if rating_scale and len(rating_scale) == matrix.shape[0]:
                ax.set_xticks(np.arange(len(rating_scale)))
                ax.set_yticks(np.arange(len(rating_scale)))
                ax.set_xticklabels(rating_scale)
                ax.set_yticklabels(rating_scale)
            
            # Rotate the x-axis tick labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations in each cell
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    # Color text based on background intensity for readability
                    text_color = 'white' if matrix[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.2f}', 
                           ha="center", va="center", color=text_color,
                           fontsize=9)
            
            # Add diagonal marker (stability pathway)
            for i in range(min(matrix.shape[0], matrix.shape[1])):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, 
                                          edgecolor='red', linewidth=2, alpha=0.7))
            
            # Add annotations if provided
            if 'annotations' in data:
                y_pos = 0.02
                for annotation in data['annotations']:
                    ax.annotate(
                        annotation, 
                        xy=(0.02, y_pos), 
                        xycoords='axes fraction',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    )
                    y_pos += 0.06  # Move up for next annotation
            
            # Adjust layout
            fig.tight_layout()
            
        return fig, ax
    
    def _plot_rating_stability(self, datasets: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Rating Stability Analysis visualization.
        
        Parameters
        ----------
        datasets : List[Dict[str, Any]]
            List of datasets to plot, each containing rating stability data
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects with the rendered plot
        """
        with self._multi_figure_context(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1.5]}) as (fig, axs):
            # We expect a single dataset for rating stability
            dataset = datasets[0]
            data = next(iter(dataset.values()))
            
            # Get data for the top subplot (period-by-period change rates)
            time_points = data.get('time_points', [])  # Time points
            period_change_rates = data.get('period_change_rates', [])  # Change rates by period
            
            # Get data for the bottom subplot (distribution of change ratios)
            change_ratios = data.get('change_ratios', [])  # Change ratios by obligor
            
            # Top plot: Period-by-period change rates
            ax1 = axs[0]
            if time_points and period_change_rates and len(time_points) == len(period_change_rates):
                ax1.plot(time_points, period_change_rates, 'o-', linewidth=2, 
                        label='Period Change Rate')
                ax1.set_xlabel('Time Period')
                ax1.set_ylabel('Change Rate')
                ax1.set_title('Rating Changes by Period')
                
                # Format x-axis ticks for time series
                if len(time_points) > 0 and isinstance(time_points[0], (str, pd.Timestamp, pd.Period)):
                    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
                
                # Add average line
                if period_change_rates:
                    avg_rate = np.mean(period_change_rates)
                    ax1.axhline(y=avg_rate, color='r', linestyle='--', 
                               label=f'Average: {avg_rate:.2f}')
                
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No period-by-period data available', 
                        ha='center', va='center', fontsize=12)
                ax1.set_xticks([])
                ax1.set_yticks([])
            
            # Bottom plot: Distribution of change ratios
            ax2 = axs[1]
            if change_ratios:
                # Create histogram of change ratios
                n, bins, patches = ax2.hist(change_ratios, bins=10, alpha=0.7, 
                                           label='Obligor Change Ratios')
                
                # Add vertical lines for key statistics
                if change_ratios:
                    mean_ratio = np.mean(change_ratios)
                    median_ratio = np.median(change_ratios)
                    ax2.axvline(x=mean_ratio, color='r', linestyle='--', 
                               label=f'Mean: {mean_ratio:.2f}')
                    ax2.axvline(x=median_ratio, color='g', linestyle=':', 
                               label=f'Median: {median_ratio:.2f}')
                
                ax2.set_xlabel('Change Ratio (Changes per Period)')
                ax2.set_ylabel('Number of Obligors')
                ax2.set_title('Distribution of Rating Changes Across Obligors')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.3)
                
                # Add obligor statistics as text
                if 'obligor_stats_summary' in data:
                    stats = data['obligor_stats_summary']
                    text = (
                        f"Obligors with no changes: {stats.get('n_obligors_with_no_changes', 0)}\n"
                        f"Obligors with one change: {stats.get('n_obligors_with_one_change', 0)}\n"
                        f"Obligors with multiple changes: {stats.get('n_obligors_with_multiple_changes', 0)}"
                    )
                    ax2.text(0.95, 0.95, text, transform=ax2.transAxes, 
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            else:
                ax2.text(0.5, 0.5, 'No change ratio data available', 
                        ha='center', va='center', fontsize=12)
                ax2.set_xticks([])
                ax2.set_yticks([])
            
            # Add annotations if provided
            if 'annotations' in data:
                y_pos = 0.02
                for annotation in data['annotations']:
                    fig.text(0.02, y_pos, annotation, fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
                    y_pos += 0.06  # Move up for next annotation
            
            # Adjust layout
            fig.tight_layout()
            
        return fig, axs[0]
        
    def _plot_concentration(self, datasets: List[Dict[str, Any]]) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot concentration analysis (Herfindahl Index) for rating grades.
        
        This creates a bar chart showing the distribution of exposures across
        rating grades, with annotations for concentration metrics.
        
        Parameters
        ----------
        datasets : List[Dict[str, Any]]
            List of dictionaries containing data for visualization
            
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes with the plotted data
        """
        with self._figure_context(figsize=(10, 6)) as (fig, ax):
            # Extract data from the first dataset
            data = next(iter(datasets[0].values()))
            
            # Extract variables needed for plotting
            x = data.get('x', [])  # Rating labels
            y = data.get('y', [])  # Proportions
            title = data.get('title', 'Rating Grade Concentration')
            xlabel = data.get('xlabel', 'Rating Grade')
            ylabel = data.get('ylabel', 'Proportion of Total Exposure')
            
            # Create bar plot of proportions
            bars = ax.bar(x, y, color='skyblue', alpha=0.8)
            
            # Highlight the bars with highest concentration
            if len(bars) > 0:
                # Highlight the top 3 most concentrated grades
                top_indices = np.argsort(y)[-3:]
                for idx in top_indices:
                    if idx < len(bars):
                        bars[idx].set_color('darkblue')
                        bars[idx].set_alpha(0.9)
            
            # Add percentage labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
            
            # Add a horizontal line for a perfectly even distribution
            if len(x) > 0:
                even_distribution = 1 / len(x)
                ax.axhline(y=even_distribution, color='red', linestyle='--', 
                          alpha=0.7, label=f'Even Distribution ({even_distribution:.1%})')
            
            # Set axis labels and title
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            
            # Add annotations if provided
            if 'annotations' in data:
                annotation_text = '\n'.join(data['annotations'])
                ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                                        fc="white", ec="gray", alpha=0.8),
                       fontsize=9)
            
            # Add grid and legend
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            ax.legend()
            
            # Rotate x-axis labels if there are many categories
            if len(x) > 5:
                plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            fig.tight_layout()
            
        return fig, ax
