"""
Regression Plotter Module
Contains the RegressionPlotter class for creating regression plots.
"""

import matplotlib.pyplot as plt
from .base_plotter import BasePlotter
from config import setup_logging
import constants

# Setup logger
logger = setup_logging(__name__)

class RegressionPlotter(BasePlotter):
    """
    Creates regression plots like actual vs. predicted and residuals vs. predicted.
    """

    def plot_actual_vs_predicted(self, y_true, y_pred, r2, intercept=None, 
                                coefficients=None, ax=None):
        """Creates actual vs predicted values scatter plot with regression equation.

        Args:
            y_true (numpy.ndarray): The true values.
            y_pred (numpy.ndarray): The predicted values.
            r2 (float): The R-squared value.
            intercept (float, optional): The intercept of the regression line. Defaults to None.
            coefficients (list, optional): The coefficients of the regression line. Defaults to None.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        """
        if ax is None:
            _, ax = self._create_figure()
        
        ax.scatter(y_true, y_pred, alpha=0.6, color='blue', s=30)
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{constants.ACTUAL_VS_PREDICTED_TITLE}\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add regression equation if provided
        if intercept is not None and coefficients is not None:
            equation = self._format_equation(intercept, coefficients)
            # Truncate long equations
            if len(equation) > 50:
                equation = equation[:47] + "..."
            ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=10, fontfamily='monospace')
        logger.info("Plotted actual vs predicted values.")

    def plot_residuals_vs_predicted(self, y_pred, residuals, ax=None):
        """Creates residuals vs predicted values plot.

        Args:
            y_pred (numpy.ndarray): The predicted values.
            residuals (numpy.ndarray): The residuals.
            ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        """
        if ax is None:
            _, ax = self._create_figure()
        
        ax.scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
        ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Line')
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title(constants.RESIDUALS_VS_PREDICTED_TITLE, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        logger.info("Plotted residuals vs predicted values.")

    def _format_equation(self, intercept, coefficients, feature_names=None):
        """Formats regression equation string.

        Args:
            intercept (float): The intercept of the regression line.
            coefficients (list): The coefficients of the regression line.
            feature_names (list, optional): The names of the features. Defaults to None.

        Returns:
            str: The formatted regression equation.
        """
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(len(coefficients))]
        
        equation = f"y = {intercept:.2f}"
        
        for i, coef in enumerate(coefficients):
            if coef >= 0:
                equation += f" + {coef:.2f}{feature_names[i]}"
            else:
                equation += f" - {abs(coef):.2f}{feature_names[i]}"
        
        return equation
