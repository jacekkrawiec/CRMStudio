"""
Centralized plotting service for CRMStudio.

This module provides a unified plotting mechanism for all metrics,
reducing code duplication and providing consistent visualization styling.
"""

import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from typing import Dict, Optional, Any, Union, List, Tuple
import pandas as pd

class PlottingService:
    """
    Centralized service for all plotting operations in CRMStudio.
    
    This class handles all visualization needs for metrics, including:
    - Curve plots (ROC, CAP)
    - Distribution plots (histograms, KS plot)
    - Group analysis plots (time series, segments)
    
    By centralizing plotting logic, we ensure consistent styling and
    reduce code duplication across metric classes.
    """
    
    def __init__(self, style: Optional[Dict] = None):
        """
        Initialize the plotting service with a style configuration.
        
        Parameters
        ----------
        style : Dict, optional
            Style configuration dictionary; if None, default style will be loaded
        """
        self.style = style or self._load_default_style()
    
    def _load_default_style(self) -> Dict:
        """
        Load default plotting style from configuration.
        
        Returns
        -------
        Dict
            Default style configuration
        """
        from .config_loader import load_config
        default_style_path = "src/crmstudio/core/templates/figure_style.yaml"
        return load_config(default_style_path)
    
    def set_style(self, style: Dict):
        """
        Update the plotting style.
        
        Parameters
        ----------
        style : Dict
            New style configuration to use
        """
        self.style = style
    
    def _convert_to_image(self, fig: plt.Figure) -> Dict[str, Any]:
        """
        Convert a matplotlib figure to a dictionary with image data.
        
        Parameters
        ----------
        fig : matplotlib.Figure
            Figure to convert
            
        Returns
        -------
        Dict
            Dictionary with image_base64, width, and height
        """
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
    
    def _detect_plot_type(self, figure_data: Dict) -> str:
        """
        Detect the type of plot needed based on figure_data.
        
        Parameters
        ----------
        figure_data : Dict
            Dictionary containing data for visualization
            
        Returns
        -------
        str
            Plot type: 'curve', 'histogram', 'distribution', or 'unknown'
        """
        if all(k in figure_data for k in ['x', 'y']):
            return 'curve'
        elif all(k in figure_data for k in ['x_def', 'y_def', 'x_ndef', 'y_ndef']):
            return 'histogram'
        elif all(k in figure_data for k in ['bin_edges']):
            return 'histogram'
        elif 'percentiles' in figure_data:
            return 'distribution'
        return 'unknown'
    
    def plot(self, figure_data: Dict, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot a single figure based on figure_data.
        
        Parameters
        ----------
        figure_data : Dict
            Dictionary containing data for visualization
        title : str, optional
            Optional title to override the one in figure_data
            
        Returns
        -------
        Dict
            Dictionary with image data
        """
        plot_type = self._detect_plot_type(figure_data)
        
        if title:
            figure_data = figure_data.copy()
            figure_data['title'] = title
        
        if plot_type == 'curve':
            return self._plot_curve(figure_data)
        elif plot_type == 'histogram':
            return self._plot_distribution(figure_data)
        else:
            # Default to curve plotting
            return self._plot_curve(figure_data)
    
    def _validate_figure_data(self, figure_data: Dict, required_keys: List[str]) -> bool:
        """
        Validate that figure_data contains all required keys.
        
        Parameters
        ----------
        figure_data : Dict
            Dictionary containing data for visualization
        required_keys : List[str]
            List of required keys to check for
            
        Returns
        -------
        bool
            True if valid, False otherwise
            
        Raises
        ------
        ValueError
            If figure_data is missing required keys
        """
        if not all(k in figure_data for k in required_keys):
            missing_keys = [k for k in required_keys if k not in figure_data]
            raise ValueError(f"Figure data is missing required keys: {missing_keys}")
        return True
            
    def _plot_curve(self, figure_data: Dict) -> Dict[str, Any]:
        """
        Plot a curve (ROC, CAP, etc.) with consistent styling.
        
        Parameters
        ----------
        figure_data : Dict
            Dictionary containing at minimum 'x' and 'y' coordinates
            
        Returns
        -------
        Dict
            Dictionary with image data
        """
        # Validate required keys
        self._validate_figure_data(figure_data, ['x', 'y'])
        
        # Extract style configurations
        colors = self.style.get('colors', {})
        fig_style = self.style.get('figure', {})
        line_style = self.style.get('lines', {})
        legend_style = self.style.get('legend', {})

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
        if self.style.get('axes', {}).get('show_diagonal', True):
            ax.plot([0, 1], [0, 1], 
                   color=colors.get('reference_line', '#ff7f0e'),
                   linestyle='--',
                   alpha=line_style.get('reference_alpha', 0.5),
                   linewidth=line_style.get('reference_width', 1.5),
                   label='Random')

        # Apply common styling
        self._apply_common_styling(ax, figure_data)

        # Set axis labels with consistent font sizes
        ax.set_xlabel(xlabel, fontsize=fig_style.get('label_fontsize', 10))
        ax.set_ylabel(ylabel, fontsize=fig_style.get('label_fontsize', 10))
        ax.set_title(title, fontsize=fig_style.get('title_fontsize', 12))
            
        # Add percentage ticks if specified
        if figure_data.get('use_percentage_ticks', False):
            step = self.style.get('axes', {}).get('percentages', {}).get('step', 20)
            ax.set_xticks(np.arange(0, 101, step))
            ax.set_xticklabels([f"{x}%" for x in range(0, 101, step)])
            if figure_data.get('use_percentage_ticks_y', False):
                ax.set_yticks(np.arange(0, 101, step))
                ax.set_yticklabels([f"{y}%" for y in range(0, 101, step)])

        # Configure legend with consistent style
        ax.legend(fontsize=legend_style.get('fontsize', 10),
                 framealpha=legend_style.get('framealpha', 0.8))
        
        return self._convert_to_image(fig)
    
    def _plot_distribution(self, figure_data: Dict) -> Dict[str, Any]:
        """
        Plot a distribution visualization with consistent styling.
        
        Parameters
        ----------
        figure_data : Dict
            Dictionary containing data for visualization
            
        Returns
        -------
        Dict
            Dictionary with image data
        """
        # Extract style configurations
        colors = self.style.get('colors', {})
        fig_style = self.style.get('figure', {})
        line_style = self.style.get('lines', {})
        legend_style = self.style.get('legend', {})
        
        # Create figure with consistent size
        fig_size = fig_style.get('size', [8, 5])
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Plot distributions if present (histogram case)
        if all(k in figure_data for k in ['x_def', 'y_def', 'x_ndef', 'y_ndef', 'bin_edges']):
            # Validate required keys
            self._validate_figure_data(figure_data, ['x_def', 'y_def', 'x_ndef', 'y_ndef', 'bin_edges'])
            ax.hist(figure_data['x_def'], bins=figure_data['bin_edges'], weights=figure_data['y_def'],
                alpha=0.5, color=colors.get('defaulted', 'red'),
                label='Defaulted')
            ax.hist(figure_data['x_ndef'], bins=figure_data['bin_edges'], weights=figure_data['y_ndef'],
                alpha=0.5, color=colors.get('non_defaulted', 'green'),
                label='Non-defaulted')
                
        # Plot main curve if present (KS, Lift, Gain case)
        elif 'x' in figure_data and 'y' in figure_data:
            # Validate required keys
            self._validate_figure_data(figure_data, ['x', 'y'])
            label = figure_data.get("actual_label", "Actual")
            ax.plot(figure_data['x'], figure_data['y'],
                    color=colors.get('main', '#1f77b4'),
                    linewidth=line_style.get('main_width', 2),
                    label=label)
        else:
            raise ValueError("Figure data does not contain either histogram data or curve data")
        
        # Plot reference curve if present
        if 'x_ref' in figure_data and 'y_ref' in figure_data:
            label = figure_data.get("ref_label", "Reference")
            ax.plot(figure_data['x_ref'], figure_data['y_ref'],
                   color=colors.get('reference', '#ff7f0e'),
                   linestyle='--',
                   alpha=line_style.get('reference_alpha', 0.5),
                   label=label)
        
        # Apply common styling
        self._apply_common_styling(ax, figure_data)
        
        # Set labels
        ax.set_xlabel(figure_data.get('xlabel', 'X'), fontsize=fig_style.get('label_fontsize', 10))
        ax.set_ylabel(figure_data.get('ylabel', 'Y'), fontsize=fig_style.get('label_fontsize', 10))
        ax.set_title(figure_data.get('title', 'Distribution'), fontsize=fig_style.get('title_fontsize', 12))

        # Add metric value if present
        if 'value' in figure_data:
            ax.text(0.95, 0.95, f"Value: {figure_data['value']:.3f}",
                   transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='bottom')

        # Configure legend with consistent style
        ax.legend(fontsize=legend_style.get('fontsize', 10),
                 framealpha=legend_style.get('framealpha', 0.8))
        
        return self._convert_to_image(fig)
    
    def _plot_multiple_curves(self, results: pd.DataFrame, group_col: str = 'group') -> Dict[str, Any]:
        """
        Plot multiple curves on the same axes.
        
        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing group results with figure_data column
        group_col : str, optional
            Column name for the group identifier
            
        Returns
        -------
        Dict
            Dictionary with image data
        """
        # Extract style configurations
        colors = self.style.get('colors', {})
        fig_style = self.style.get('figure', {})
        grid_style = self.style.get('grid', {})
        line_style = self.style.get('lines', {})
        legend_style = self.style.get('legend', {})
        
        # Create figure
        fig_size = fig_style.get('size', [10, 6])
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Plot each curve
        for _, row in results.iterrows():
            group = row[group_col]
            figure_data = row['figure_data']
            ax.plot(figure_data['x'], figure_data['y'], 
                   label=f"{group}",
                   alpha=line_style.get('main_alpha', 0.7))
        
        # Set labels from first figure_data
        first_figure = results.iloc[0]['figure_data']
        ax.set_xlabel(first_figure.get('xlabel', 'X'), fontsize=fig_style.get('label_fontsize', 10))
        ax.set_ylabel(first_figure.get('ylabel', 'Y'), fontsize=fig_style.get('label_fontsize', 10))
        
        # Add grid if specified
        if grid_style.get('show', True):
            ax.grid(True, 
                   linestyle=grid_style.get('linestyle', '--'),
                   alpha=grid_style.get('alpha', 0.3),
                   color=grid_style.get('color', '#cccccc'))
        
        # Configure legend
        ax.legend(fontsize=legend_style.get('fontsize', 10),
                 framealpha=legend_style.get('framealpha', 0.8))
        
        return self._convert_to_image(fig)
    
    def _plot_multiple_distributions(self, results: pd.DataFrame, group_col: str = 'group') -> Dict[str, Any]:
        """
        Plot multiple distributions as a grid of subplots.
        
        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing group results with figure_data column
        group_col : str, optional
            Column name for the group identifier
            
        Returns
        -------
        Dict
            Dictionary with image data
        """
        # Calculate grid dimensions
        nrows = (len(results) + 1) // 2
        
        # Create figure with subplots
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=2,
            figsize=(15, 5 * nrows)
        )
        axes = axes.flatten()
        
        # For each group, plot its distribution
        for idx, (_, row) in enumerate(results.iterrows()):
            ax = axes[idx]
            group = row[group_col]
            figure_data = row['figure_data']
            
            # Plot distributions if present
            if all(k in figure_data for k in ['x_def', 'y_def', 'x_ndef', 'y_ndef', 'bin_edges']):
                self._add_histogram_to_axis(ax, figure_data)
            
            # Plot curves if present
            elif 'x' in figure_data and 'y' in figure_data:
                self._add_curve_to_axis(ax, figure_data)
            
            # Set title with group info
            ax.set_title(f"{group} (n={figure_data.get('n_obs', '')})")
        
        # Hide any unused subplots
        for idx in range(len(results), len(axes)):
            axes[idx].set_visible(False)
        
        # Add overall title if needed
        # fig.suptitle("Group Analysis", fontsize=16)
        
        # Adjust layout
        fig.tight_layout()
        
        return self._convert_to_image(fig)
    
    def _plot_group_analysis(self, results: pd.DataFrame, metric_type: str = 'curve') -> Dict[str, Any]:
        """
        Plot analysis results by group.
        
        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing group results
        metric_type : str
            Type of metric: 'curve' or 'distribution'
            
        Returns
        -------
        Dict
            Dictionary with image data
        """
        has_figure_data = 'figure_data' in results.columns
        if has_figure_data:
            if results['figure_data'].isna().all():
                has_figure_data = False
        
        if has_figure_data:
            if metric_type == 'curve':
                return self._plot_multiple_curves(results)
            else:
                return self._plot_multiple_distributions(results)
        else:
            return self._plot_group_values(results)
    
    def _plot_group_values(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        Plot metric values across groups.
        
        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing group results with 'value' column
            
        Returns
        -------
        Dict
            Dictionary with image data
        """
        # Extract style configurations
        colors = self.style.get('colors', {})
        fig_style = self.style.get('figure', {})
        grid_style = self.style.get('grid', {})
        line_style = self.style.get('lines', {})
        
        # Create figure
        fig_size = fig_style.get('size', [10, 6])
        fig, ax = plt.subplots(figsize=fig_size)
        
        group_type = results['group_type'].iloc[0]
        
        if group_type == 'time':
            # Time series plot
            x_values = [str(p) for p in results['group']]
            ax.plot(x_values, results['value'],
                   marker='o',
                   color=colors.get('main', '#1f77b4'),
                   linewidth=line_style.get('main_width', 2))
            plt.xticks(rotation=45)
            ax.set_xlabel('Period', fontsize=fig_style.get('label_fontsize', 10))
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
            ax.set_xlabel(results['group_name'].iloc[0].title(), 
                         fontsize=fig_style.get('label_fontsize', 10))
        
        ax.set_ylabel("Metric Value", fontsize=fig_style.get('label_fontsize', 10))
        
        # Add sample size as secondary axis
        ax2 = ax.twinx()
        ax2.plot(range(len(results)), results['n_obs'],
                color='gray',
                linestyle=':',
                alpha=0.6,
                label='Sample Size')
        ax2.set_ylabel('Sample Size', fontsize=fig_style.get('label_fontsize', 10))
        
        # Add grid if specified
        if grid_style.get('show', True):
            ax.grid(True, 
                   linestyle=grid_style.get('linestyle', '--'),
                   alpha=grid_style.get('alpha', 0.3),
                   color=grid_style.get('color', '#cccccc'))
        
        return self._convert_to_image(fig)
    
    def _add_histogram_to_axis(self, ax, figure_data: Dict):
        """
        Helper method to add histogram visualization to an axis.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to add the histogram to
        figure_data : Dict
            Dictionary containing histogram data
            
        Raises
        ------
        ValueError
            If figure_data is missing required keys
        """
        # Validate required keys
        self._validate_figure_data(figure_data, ['x_def', 'y_def', 'x_ndef', 'y_ndef', 'bin_edges'])
        
        colors = self.style.get('colors', {})
        fig_style = self.style.get('figure', {})
        
        bin_edges = figure_data['bin_edges']
        
        # Plot histograms
        ax.hist(figure_data['x_def'], bins=bin_edges, weights=figure_data['y_def'],
               alpha=0.5, color=colors.get('defaulted', 'red'),
               label='Defaulted')
        ax.hist(figure_data['x_ndef'], bins=bin_edges, weights=figure_data['y_ndef'],
               alpha=0.5, color=colors.get('non_defaulted', 'green'),
               label='Non-defaulted')
        
        # Apply common styling
        self._apply_common_styling(ax, figure_data)
        
        # Add labels
        ax.set_xlabel(figure_data.get('xlabel', 'Score'), fontsize=fig_style.get('label_fontsize', 10))
        ax.set_ylabel(figure_data.get('ylabel', 'Frequency'), fontsize=fig_style.get('label_fontsize', 10))
        ax.legend()
    
    def _add_curve_to_axis(self, ax, figure_data: Dict):
        """
        Helper method to add curve visualization to an axis.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to add the curve to
        figure_data : Dict
            Dictionary containing curve data
            
        Raises
        ------
        ValueError
            If figure_data is missing required keys
        """
        # Validate required keys
        self._validate_figure_data(figure_data, ['x', 'y'])
        
        colors = self.style.get('colors', {})
        line_style = self.style.get('lines', {})
        fig_style = self.style.get('figure', {})
        
        # Plot main curves
        label = figure_data.get("actual_label", "Actual")
        ax.plot(figure_data['x'], figure_data['y'],
               color=colors.get('main', '#1f77b4'),
               linewidth=line_style.get('main_width', 2),
               label=label)
        
        # Add reference line if present
        if 'x_ref' in figure_data and 'y_ref' in figure_data:
            label = figure_data.get("ref_label", "Reference")
            ax.plot(figure_data['x_ref'], figure_data['y_ref'],
                   color=colors.get('reference', '#ff7f0e'),
                   linestyle='--',
                   alpha=line_style.get('reference_alpha', 0.5),
                   label=label)
        
        # Apply common styling
        self._apply_common_styling(ax, figure_data)
        
        # Add labels
        ax.set_xlabel(figure_data.get('xlabel', 'X'), fontsize=fig_style.get('label_fontsize', 10))
        ax.set_ylabel(figure_data.get('ylabel', 'Y'), fontsize=fig_style.get('label_fontsize', 10))
        
        # Add metric value if present
        if 'value' in figure_data:
            ax.text(0.95, 0.05,
                   f"Value: {figure_data['value']:.3f}",
                   transform=ax.transAxes,
                   horizontalalignment='right',
                   verticalalignment='bottom')
        
        ax.legend()
    
    def _apply_common_styling(self, ax, figure_data: Dict = None):
        """
        Apply common styling elements to an axis.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis to apply styling to
        figure_data : Dict, optional
            Optional figure data with additional styling info
        """
        fig_style = self.style.get('figure', {})
        grid_style = self.style.get('grid', {})
        
        # Apply grid if specified
        if grid_style.get('show', True):
            ax.grid(True,
                   linestyle=grid_style.get('linestyle', '--'),
                   alpha=grid_style.get('alpha', 0.3),
                   color=grid_style.get('color', '#cccccc'))
        
        # Set tick label sizes
        ax.tick_params(labelsize=fig_style.get('tick_fontsize', 8))
        
        # Set axis limits if provided in figure_data
        if figure_data:
            if 'xlim' in figure_data:
                ax.set_xlim(figure_data['xlim'])
            if 'ylim' in figure_data:
                ax.set_ylim(figure_data['ylim'])
    
    def display_image(self, image_data: Dict):
        """
        Display an image in the current context (notebook or matplotlib window).
        
        Parameters
        ----------
        image_data : Dict
            Dictionary with image_base64 key
        """
        img_data = base64.b64decode(image_data["image_base64"])
        img = plt.imread(BytesIO(img_data))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def save_image(self, image_data: Dict, filepath: str):
        """
        Save an image to disk.
        
        Parameters
        ----------
        image_data : Dict
            Dictionary with image_base64 key
        filepath : str
            Path where to save the image
        """
        img_data = base64.b64decode(image_data["image_base64"])
        with open(filepath, "wb") as f:
            f.write(img_data)
