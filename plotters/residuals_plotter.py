"""
Residuals Plotter Module
Contains the ResidualsPlotter class for creating residuals analysis plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from .base_plotter import BasePlotter
from config import setup_logging
import constants

# Setup logger
logger = setup_logging(__name__)

class ResidualsPlotter(BasePlotter):
    """
    Creates residuals analysis plots like histogram and Q-Q plot.
    """

    def plot_residual_histogram(self, residuals, ax=None):
        """Creates histogram of residuals with normal distribution overlay.

        Args:
            residuals (numpy.ndarray): The residuals.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        """
        if ax is None:
            _, ax = self._create_figure()
        
        ax.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black', 
               density=True, label='Residuals')
        
        # Overlay normal distribution
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
               label=f'Normal(mu={mu:.2f}, sigma={sigma:.2f})')
        
        ax.set_xlabel('Residuals', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(constants.RESIDUAL_HISTOGRAM_TITLE, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        logger.info("Plotted residual histogram.")

    def plot_qq_plot(self, residuals, ax=None):
        """Creates Q-Q plot for residual normality check.

        Args:
            residuals (numpy.ndarray): The residuals.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        """
        if ax is None:
            _, ax = self._create_figure()
        
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(constants.QQ_PLOT_TITLE, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        logger.info("Plotted Q-Q plot.")
