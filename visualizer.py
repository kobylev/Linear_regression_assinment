"""
Visualization Facade Module
Provides a simplified interface for creating all plots.
"""

from plotters import (
    AnalysisPlotter,
    CoefficientPlotter,
    RegressionPlotter,
    ResidualsPlotter
)
from config import setup_logging
import constants

# Setup logger
logger = setup_logging(__name__)

class Visualizer:
    """
    Facade for creating all visualizations.
    """
    def __init__(self, figsize=(15, 12), style='whitegrid'):
        self.analysis_plotter = AnalysisPlotter(figsize=figsize, style=style)
        self.coef_plotter = CoefficientPlotter(figsize=figsize, style=style)
        self.reg_plotter = RegressionPlotter(figsize=figsize, style=style)
        self.res_plotter = ResidualsPlotter(figsize=figsize, style=style)
        logger.info("Initialized Visualizer facade.")

    def create_comprehensive_plot(self, y_true, y_pred, residuals, data, 
                                true_coef, fitted_coef, r2, intercept=None):
        fig = self.analysis_plotter.create_comprehensive_plot(
            y_true, y_pred, residuals, data, true_coef, fitted_coef, r2, intercept
        )
        self.analysis_plotter.save_plot(fig, constants.COMPREHENSIVE_PLOT_FILENAME)

    def plot_sample_size_analysis(self, sample_sizes, r2_values, r2_adj_values):
        fig = self.analysis_plotter.plot_sample_size_analysis(
            sample_sizes, r2_values, r2_adj_values
        )
        self.analysis_plotter.save_plot(fig, constants.SAMPLE_SIZE_PLOT_FILENAME)
