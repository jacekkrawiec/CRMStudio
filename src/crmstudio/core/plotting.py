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
            The type of plot ('curve', 'distribution', 'calibration'). If None, will be detected from data.
            
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
                        if data.get('plot_type') != 'hosmer_lemeshow':
                            ax.set_xlim([0, 1])
                            ax.set_ylim([0, 1])
                    
                    # Use percentage ticks if specified
                    if data.get('use_percentage_ticks', False):
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
                    if data.get('use_percentage_ticks_y', False):
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                    
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
