"""
Visualization Module for Linear Regression Analysis
Handles all plotting and visualization tasks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

class RegressionVisualizer:
    """
    Creates various plots for regression analysis.
    """
    
    def __init__(self, figsize=(15, 12), style='whitegrid'):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size
        style : str
            Seaborn style
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_style(style)
        sns.set_palette("husl")
    
    def plot_actual_vs_predicted(self, y_true, y_pred, r2, ax=None):
        """
        Create actual vs predicted values scatter plot.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True values
        y_pred : numpy.ndarray
            Predicted values
        r2 : float
            R² value
        ax : matplotlib.axes, optional
            Axes object to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_true, y_pred, alpha=0.6, color='blue', s=30)
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Actual vs Predicted\nR² = {r2:.4f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def plot_residuals_vs_predicted(self, y_pred, residuals, ax=None):
        """
        Create residuals vs predicted values plot.
        
        Parameters:
        -----------
        y_pred : numpy.ndarray
            Predicted values
        residuals : numpy.ndarray
            Residuals
        ax : matplotlib.axes, optional
            Axes object to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted Values')
        ax.grid(True, alpha=0.3)
    
    def plot_residual_histogram(self, residuals, ax=None):
        """
        Create histogram of residuals.
        
        Parameters:
        -----------
        residuals : numpy.ndarray
            Residuals
        ax : matplotlib.axes, optional
            Axes object to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black', density=True)
        
        # Overlay normal distribution
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Residuals')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_correlation_heatmap(self, data, ax=None):
        """
        Create correlation heatmap of features.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Dataset containing features and target
        ax : matplotlib.axes, optional
            Axes object to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        corr_matrix = data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', ax=ax)
        ax.set_title('Feature Correlation Matrix')
    
    def plot_coefficients_comparison(self, true_coef, fitted_coef, ax=None):
        """
        Compare true vs fitted coefficients.
        
        Parameters:
        -----------
        true_coef : list
            True coefficients
        fitted_coef : list
            Fitted coefficients
        ax : matplotlib.axes, optional
            Axes object to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos = np.arange(len(true_coef))
        width = 0.35
        
        ax.bar(x_pos - width/2, true_coef, width, label='True Coefficients', alpha=0.7, color='skyblue')
        ax.bar(x_pos + width/2, fitted_coef, width, label='Fitted Coefficients', alpha=0.7, color='lightcoral')
        
        ax.set_xlabel('Coefficient Index')
        ax.set_ylabel('Coefficient Value')
        ax.set_title('True vs Fitted Coefficients')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'β{i+1}' for i in range(len(true_coef))])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_qq_plot(self, residuals, ax=None):
        """
        Create Q-Q plot for residual normality check.
        
        Parameters:
        -----------
        residuals : numpy.ndarray
            Residuals
        ax : matplotlib.axes, optional
            Axes object to plot on
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals\n(Normality Check)')
        ax.grid(True, alpha=0.3)
    
    def create_comprehensive_plot(self, y_true, y_pred, residuals, data, 
                                true_coef, fitted_coef, r2):
        """
        Create comprehensive analysis plot with multiple subplots.
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True values
        y_pred : numpy.ndarray
            Predicted values
        residuals : numpy.ndarray
            Residuals
        data : pandas.DataFrame
            Dataset
        true_coef : list
            True coefficients
        fitted_coef : list
            Fitted coefficients
        r2 : float
            R² value
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Plot 1: Actual vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        self.plot_actual_vs_predicted(y_true, y_pred, r2, ax1)
        
        # Plot 2: Residuals vs Predicted
        ax2 = plt.subplot(2, 3, 2)
        self.plot_residuals_vs_predicted(y_pred, residuals, ax2)
        
        # Plot 3: Residual histogram
        ax3 = plt.subplot(2, 3, 3)
        self.plot_residual_histogram(residuals, ax3)
        
        # Plot 4: Correlation heatmap
        ax4 = plt.subplot(2, 3, 4)
        self.plot_correlation_heatmap(data, ax4)
        
        # Plot 5: Coefficients comparison
        ax5 = plt.subplot(2, 3, 5)
        self.plot_coefficients_comparison(true_coef, fitted_coef, ax5)
        
        # Plot 6: Q-Q plot
        ax6 = plt.subplot(2, 3, 6)
        self.plot_qq_plot(residuals, ax6)
        
        plt.tight_layout()
        return fig
    
    def plot_sample_size_analysis(self, sample_sizes, r2_values, r2_adj_values):
        """
        Plot the effect of sample size on R² vs Adjusted R².
        
        Parameters:
        -----------
        sample_sizes : list
            Different sample sizes tested
        r2_values : list
            R² values for each sample size
        r2_adj_values : list
            Adjusted R² values for each sample size
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sample_sizes, r2_values, 'bo-', label='R²', linewidth=2, markersize=8)
        ax.plot(sample_sizes, r2_adj_values, 'ro-', label='Adjusted R²', linewidth=2, markersize=8)
        
        ax.set_xlabel('Sample Size (n)')
        ax.set_ylabel('R² Value')
        ax.set_title('R² vs Adjusted R² as Function of Sample Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        return fig