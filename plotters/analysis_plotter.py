"""
Analysis Plotter Module
Contains the AnalysisPlotter class for creating comprehensive analysis plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from .base_plotter import BasePlotter
from .regression_plotter import RegressionPlotter
from .residuals_plotter import ResidualsPlotter
from .coefficient_plotter import CoefficientPlotter
from config import setup_logging
import constants

# Setup logger
logger = setup_logging(__name__)

class AnalysisPlotter(BasePlotter):
    """
    Creates comprehensive analysis plots for regression models.
    """

    def create_comprehensive_plot(self, y_true, y_pred, residuals, data, 
                                true_coef, fitted_coef, r2, intercept=None):
        """Creates a comprehensive analysis plot with multiple subplots.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.
            residuals (numpy.ndarray): The residuals.
            data (pandas.DataFrame): The dataframe containing the data.
            true_coef (list): The true coefficients.
            fitted_coef (list): The fitted coefficients.
            r2 (float): The R-squared value.
            intercept (float, optional): The intercept of the regression line. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The comprehensive analysis plot.
        """
        fig, axes = self._create_figure(num_subplots=6, nrows=2, ncols=3)
        fig.suptitle(f'{constants.COMPREHENSIVE_PLOT_SUPER_TITLE} - R² = {r2:.4f}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        reg_plotter = RegressionPlotter()
        res_plotter = ResidualsPlotter()
        coef_plotter = CoefficientPlotter()

        reg_plotter.plot_actual_vs_predicted(y_true, y_pred, r2, intercept, fitted_coef, ax=axes[0])
        reg_plotter.plot_residuals_vs_predicted(y_pred, residuals, ax=axes[1])
        res_plotter.plot_residual_histogram(residuals, ax=axes[2])
        coef_plotter.plot_correlation_heatmap(data, ax=axes[3])
        coef_plotter.plot_coefficients_comparison(true_coef, fitted_coef, ax=axes[4])
        res_plotter.plot_qq_plot(residuals, ax=axes[5])
        
        plt.tight_layout()
        logger.info("Created comprehensive analysis plot.")
        return fig

    def plot_sample_size_analysis(self, sample_sizes, r2_values, r2_adj_values):
        """Plots the effect of sample size on R² vs Adjusted R² with improved scaling.

        Args:
            sample_sizes (list): The list of sample sizes.
            r2_values (list): The list of R-squared values.
            r2_adj_values (list): The list of adjusted R-squared values.

        Returns:
            matplotlib.figure.Figure: The sample size analysis plot.
        """
        fig, ax = self._create_figure()
        
        sample_sizes = np.array(sample_sizes)
        r2_values = np.array(r2_values)
        r2_adj_values = np.array(r2_adj_values)
        
        ax.plot(sample_sizes, r2_values, 'bo-', label='R²', linewidth=3, markersize=10)
        ax.plot(sample_sizes, r2_adj_values, 'ro-', label='Adjusted R²', linewidth=3, markersize=10)
        
        ax.fill_between(sample_sizes, r2_values, r2_adj_values, alpha=0.3, color='gray', 
                       label='Adjustment Penalty')
        
        ax.set_xlabel('Sample Size (n)', fontsize=14)
        ax.set_ylabel('R² Value', fontsize=14)
        ax.set_title(constants.SAMPLE_SIZE_ANALYSIS_TITLE, fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        y_min = min(r2_adj_values.min(), r2_values.min())
        y_max = max(r2_adj_values.max(), r2_values.max())
        y_range = y_max - y_min
        
        if y_range < 0.001:
            y_center = (y_min + y_max) / 2
            y_min = y_center - 0.0005
            y_max = y_center + 0.0005
        else:
            padding = y_range * 0.1
            y_min -= padding
            y_max += padding
        
        ax.set_ylim(y_min, y_max)
        logger.info("Plotted sample size analysis.")
        return fig
