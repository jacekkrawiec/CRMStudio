"""
Centralized plotting service for CRMStudio.

This module provides a unified plotting mechanism for all metrics,
reducing code duplication and providing consistent visualization styling.
"""

import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from typing import Dict, Optional, Any, List
import pandas as pd

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
    
    def _convert_to_image(self, fig: plt.Figure) -> Dict[str, Any]:
        """Convert a matplotlib figure to a dictionary with image data."""
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
    
    def plot(self, figure_data: Dict, plot_type: str = None) -> Dict[str, Any]:
        """
        Plot a figure based on figure_data and the specified plot type.
        
        Parameters
        ----------
        figure_data : Dict
            Dictionary containing data for visualization
        plot_type : str, optional
            The type of plot ('curve' or 'distribution'). If None, will be detected from data.
            
        Returns
        -------
        Dict
            Dictionary with image data
        """
        # Create figure with consistent size
        fig_style = self.style.get('figure', {})
        fig_size = fig_style.get('size', [8, 5])
        fig, ax = plt.subplots(figsize=fig_size)
        
        # Route to appropriate plotting function based on plot_type
        if plot_type == 'curve' or (plot_type is None and all(k in figure_data for k in ['x', 'y'])):
            self._plot_curve_data(ax, figure_data)
        else:  # distribution
            self._plot_distribution_data(ax, figure_data)
        
        # Apply common styling and settings
        self._apply_common_styling(ax, figure_data)
        
        # Return the image data
        return self._convert_to_image(fig)
    
    def _plot_curve_data(self, ax, figure_data: Dict):
        """Plot curve data on the given axis."""
        # Extract styling options
        colors = self.style.get('colors', {})
        fig_style = self.style.get('figure', {})
        line_style = self.style.get('lines', {})
        
        # Plot main curve
        ax.plot(
            figure_data['x'], 
            figure_data['y'],
            color=colors.get('main', '#1f77b4'),
            alpha=line_style.get('main_alpha', 0.8),
            linewidth=line_style.get('main_width', 2),
            label=figure_data.get('label', 'Model')
        )
        
        # Add reference line if needed
        if figure_data.get('show_diagonal', True):
            ax.plot(
                [0, 1], 
                [0, 1],
                color=colors.get('reference', '#ff7f0e'),
                linestyle='--',
                alpha=line_style.get('reference_alpha', 0.5),
                linewidth=line_style.get('reference_width', 1.5),
                label=figure_data.get('reference_label', 'Random')
            )
        
        # Add labels
        ax.set_xlabel(figure_data.get('xlabel', 'Score'), fontsize=fig_style.get('label_fontsize', 10))
        ax.set_ylabel(figure_data.get('ylabel', 'Performance'), fontsize=fig_style.get('label_fontsize', 10))
        ax.set_title(figure_data.get('title', 'Model Performance'), fontsize=fig_style.get('title_fontsize', 12))
        
        # Configure legend
        ax.legend(fontsize=self.style.get('legend', {}).get('fontsize', 10))
    
    def _plot_distribution_data(self, ax, figure_data: Dict):
        """Plot distribution data on the given axis."""
        # Extract styling options
        colors = self.style.get('colors', {})
        fig_style = self.style.get('figure', {})
        
        # Plot defaulted/non-defaulted distributions if available
        if all(k in figure_data for k in ['x_def', 'y_def', 'x_ndef', 'y_ndef']):
            bin_edges = figure_data.get('bin_edges', None)
            if bin_edges is None:
                # Create reasonable bins if not provided
                all_x = np.concatenate([figure_data['x_def'], figure_data['x_ndef']])
                bin_edges = np.linspace(np.min(all_x), np.max(all_x), 20)
            
            # Plot histograms
            ax.hist(
                figure_data['x_def'], 
                bins=bin_edges, 
                weights=figure_data['y_def'],
                alpha=0.5, 
                color=colors.get('defaulted', 'red'),
                label='Defaulted'
            )
            ax.hist(
                figure_data['x_ndef'], 
                bins=bin_edges, 
                weights=figure_data['y_ndef'],
                alpha=0.5, 
                color=colors.get('non_defaulted', 'green'),
                label='Non-defaulted'
            )
        # Plot standard distribution curve if available
        elif all(k in figure_data for k in ['x', 'y']):
            ax.plot(
                figure_data['x'], 
                figure_data['y'],
                color=colors.get('main', '#1f77b4'),
                label=figure_data.get('label', 'Distribution')
            )
            
        # Add labels
        ax.set_xlabel(figure_data.get('xlabel', 'Value'), fontsize=fig_style.get('label_fontsize', 10))
        ax.set_ylabel(figure_data.get('ylabel', 'Frequency'), fontsize=fig_style.get('label_fontsize', 10))
        ax.set_title(figure_data.get('title', 'Distribution'), fontsize=fig_style.get('title_fontsize', 12))
        
        # Add metric value if present
        if 'value' in figure_data:
            ax.text(
                0.95, 
                0.95, 
                f"Value: {figure_data['value']:.3f}",
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top'
            )
        
        # Configure legend
        ax.legend(fontsize=self.style.get('legend', {}).get('fontsize', 10))
    
    def _apply_common_styling(self, ax, figure_data: Dict = None):
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
        
        # Set axis limits if provided
        if figure_data:
            if 'xlim' in figure_data:
                ax.set_xlim(figure_data['xlim'])
            if 'ylim' in figure_data:
                ax.set_ylim(figure_data['ylim'])
    
    def display_image(self, image_data: Dict):
        """Display an image in the current context."""
        img_data = base64.b64decode(image_data["image_base64"])
        img = plt.imread(BytesIO(img_data))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    def save_image(self, image_data: Dict, filepath: str):
        """Save an image to disk."""
        img_data = base64.b64decode(image_data["image_base64"])
        with open(filepath, "wb") as f:
            f.write(img_data)
            
    def _plot_group_analysis(self, results: pd.DataFrame, plot_type: str = 'curve') -> Dict[str, Any]:
        """
        Plot metric results across groups.
        
        Parameters
        ----------
        results : pd.DataFrame
            DataFrame containing group results
        plot_type : str
            Type of metric plot ('curve' or 'distribution')
            
        Returns
        -------
        Dict
            Dictionary with image data
        """
        # Check if figure_data is available for plotting multiple curves/distributions
        has_figure_data = 'figure_data' in results.columns and not results['figure_data'].isna().all()
        
        # Create figure
        fig_style = self.style.get('figure', {})
        fig_size = fig_style.get('size', [10, 6])
        fig, ax = plt.subplots(figsize=fig_size)
        
        if has_figure_data and plot_type == 'curve':
            # Plot multiple curves on the same axes
            for _, row in results.iterrows():
                group = row['group']
                figure_data = row['figure_data']
                if figure_data and 'x' in figure_data and 'y' in figure_data:
                    ax.plot(figure_data['x'], figure_data['y'], label=f"{group}")
            
            # Add labels from first figure_data
            first_figure = next((row['figure_data'] for _, row in results.iterrows() 
                               if row['figure_data'] and 'x' in row['figure_data']), {})
            
            ax.set_xlabel(first_figure.get('xlabel', 'X'), fontsize=fig_style.get('label_fontsize', 10))
            ax.set_ylabel(first_figure.get('ylabel', 'Y'), fontsize=fig_style.get('label_fontsize', 10))
            
        elif has_figure_data and plot_type == 'distribution':
            # For distributions, we'll show a simple value plot and leave the detailed
            # distribution visualization for individual plots
            ax.plot(results['group'], results['value'], marker='o')
            ax.set_xlabel(results['group_name'].iloc[0].title(), fontsize=fig_style.get('label_fontsize', 10))
            ax.set_ylabel('Metric Value', fontsize=fig_style.get('label_fontsize', 10))
            
            if 'group_type' in results.columns and results['group_type'].iloc[0] == 'time':
                plt.xticks(rotation=45)
                
        else:
            # Plot simple metric values with markers
            if 'group_type' in results.columns and results['group_type'].iloc[0] == 'time':
                # Time series plot
                ax.plot(results['group'], results['value'], marker='o')
                plt.xticks(rotation=45)
                ax.set_xlabel('Period', fontsize=fig_style.get('label_fontsize', 10))
            else:
                # Bar plot for segments
                x = range(len(results))
                ax.bar(x, results['value'], alpha=0.7)
                plt.xticks(x, results['group'], rotation=45)
                ax.set_xlabel(results['group_name'].iloc[0].title(), fontsize=fig_style.get('label_fontsize', 10))
                
            ax.set_ylabel('Metric Value', fontsize=fig_style.get('label_fontsize', 10))
        
        # Apply common styling
        self._apply_common_styling(ax)
        
        # Add legend if we have multiple curves
        if has_figure_data and plot_type == 'curve':
            ax.legend()
            
        # Set title
        ax.set_title(f"Group Analysis: {results['group_name'].iloc[0].title()}", 
                    fontsize=fig_style.get('title_fontsize', 12))
        
        return self._convert_to_image(fig)
