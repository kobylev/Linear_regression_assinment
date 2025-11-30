"""
Coefficient Plotter Module
Contains the CoefficientPlotter class for creating coefficient-related plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from .base_plotter import BasePlotter
from config import setup_logging
import constants

# Setup logger
logger = setup_logging(__name__)

class CoefficientPlotter(BasePlotter):
    """
    Creates coefficient-related plots like correlation heatmap and coefficient comparison.
    """

    def plot_correlation_heatmap(self, data, ax=None):
        """Creates correlation heatmap of features.

        Args:
            data (pandas.DataFrame): The dataframe containing the data.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        """
        if ax is None:
            _, ax = self._create_figure()
        
        corr_matrix = data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(constants.CORRELATION_HEATMAP_TITLE, fontsize=14, fontweight='bold')
        logger.info("Plotted correlation heatmap.")

    def plot_coefficients_comparison(self, true_coef, fitted_coef, ax=None):
        """Compares true vs fitted coefficients.

        Args:
            true_coef (list): The true coefficients.
            fitted_coef (list): The fitted coefficients.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        """
        if ax is None:
            _, ax = self._create_figure()
        
        x_pos = np.arange(len(true_coef))
        width = 0.35
        
        ax.bar(x_pos - width/2, true_coef, width, label='True Coefficients', 
               alpha=0.7, color='skyblue', edgecolor='black')
        ax.bar(x_pos + width/2, fitted_coef, width, label='Fitted Coefficients', 
               alpha=0.7, color='lightcoral', edgecolor='black')
        
        ax.set_xlabel('Coefficient Index', fontsize=12)
        ax.set_ylabel('Coefficient Value', fontsize=12)
        ax.set_title(constants.COEFFICIENTS_COMPARISON_TITLE, fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'beta{i+1}' for i in range(len(true_coef))])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        logger.info("Plotted coefficients comparison.")
