"""
Fixed Visualization Module for Linear Regression Analysis
Resolves Unicode font issues and plot scaling problems.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

class RegressionVisualizer:
    """
    Creates various plots for regression analysis with enhanced features.
    Fixed for font compatibility and proper scaling.
    """
    
    def __init__(self, figsize=(15, 12), style='whitegrid'):
        """
        Initialize the visualizer with font fixes.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size
        style : str
            Seaborn style
        """
        self.figsize = figsize
        
        # Set matplotlib to use fonts that support standard characters
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False  # Use ASCII minus sign
        
        plt.style.use('default')
        sns.set_style(style)
        sns.set_palette("husl")
    
    def format_equation(self, intercept, coefficients, feature_names=None):
        """
        Format regression equation string.
        
        Parameters:
        -----------
        intercept : float
            Intercept term
        coefficients : list
            Coefficient values
        feature_names : list, optional
            Names of features
        
        Returns:
        --------
        str
            Formatted equation string
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
    
    def plot_actual_vs_predicted(self, y_true, y_pred, r2, intercept=None, 
                                coefficients=None, ax=None):
        """
        Create actual vs predicted values scatter plot with regression equation.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_true, y_pred, alpha=0.6, color='blue', s=30)
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'Actual vs Predicted\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add regression equation if provided
        if intercept is not None and coefficients is not None:
            equation = self.format_equation(intercept, coefficients)
            # Truncate long equations
            if len(equation) > 50:
                equation = equation[:47] + "..."
            ax.text(0.05, 0.95, equation, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    def plot_residuals_vs_predicted(self, y_pred, residuals, ax=None):
        """
        Create residuals vs predicted values plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
        ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Line')
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add residual statistics
        residual_std = np.std(residuals)
        ax.text(0.05, 0.95, f'sigma_residuals = {residual_std:.2f}', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=10)
    
    def plot_residual_histogram(self, residuals, ax=None):
        """
        Create histogram of residuals with normal distribution overlay.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black', 
               density=True, label='Residuals')
        
        # Overlay normal distribution
        mu, sigma = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, 
               label=f'Normal(mu={mu:.2f}, sigma={sigma:.2f})')
        
        ax.set_xlabel('Residuals', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add normality test result - using ASCII characters only
        try:
            from scipy.stats import shapiro
            _, p_value = shapiro(residuals)
            normality_text = f'Shapiro-Wilk p = {p_value:.3f}'
            if p_value > 0.05:
                normality_text += ' (Normal OK)'
            else:
                normality_text += ' (Non-normal)'
        except:
            normality_text = 'Normality: Visual inspection'
        
        ax.text(0.05, 0.95, normality_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=10)
    
    def plot_correlation_heatmap(self, data, ax=None):
        """
        Create correlation heatmap of features.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        corr_matrix = data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Add interpretation guide
        ax.text(1.02, 0.5, 'Correlation Scale:\n|r| >= 0.7: Strong\n|r| >= 0.3: Moderate\n|r| < 0.3: Weak',
               transform=ax.transAxes, verticalalignment='center', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    def plot_coefficients_comparison(self, true_coef, fitted_coef, ax=None):
        """
        Compare true vs fitted coefficients.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos = np.arange(len(true_coef))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, true_coef, width, label='True Coefficients', 
                      alpha=0.7, color='skyblue', edgecolor='black')
        bars2 = ax.bar(x_pos + width/2, fitted_coef, width, label='Fitted Coefficients', 
                      alpha=0.7, color='lightcoral', edgecolor='black')
        
        ax.set_xlabel('Coefficient Index', fontsize=12)
        ax.set_ylabel('Coefficient Value', fontsize=12)
        ax.set_title('True vs Fitted Coefficients', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'beta{i+1}' for i in range(len(true_coef))])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar1, bar2, true_val, fitted_val in zip(bars1, bars2, true_coef, fitted_coef):
            height1, height2 = bar1.get_height(), bar2.get_height()
            ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01*abs(height1),
                   f'{true_val:.2f}', ha='center', va='bottom', fontsize=8)
            ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01*abs(height2),
                   f'{fitted_val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Add error summary
        errors = [abs(t - f) for t, f in zip(true_coef, fitted_coef)]
        max_error = max(errors)
        mean_error = np.mean(errors)
        ax.text(0.05, 0.95, f'Max Error: {max_error:.3f}\nMean Error: {mean_error:.3f}', 
               transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=10)
    
    def plot_qq_plot(self, residuals, ax=None):
        """
        Create Q-Q plot for residual normality check.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals\n(Normality Check)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Calculate R² for Q-Q plot
        try:
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sample_quantiles = np.sort(residuals)
            qq_r2 = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
            
            ax.text(0.05, 0.95, f'Q-Q R² = {qq_r2:.3f}\n(Closer to 1 = More Normal)', 
                   transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=10)
        except:
            pass  # Skip if calculation fails
    
    def create_comprehensive_plot(self, y_true, y_pred, residuals, data, 
                                true_coef, fitted_coef, r2, intercept=None):
        """
        Create comprehensive analysis plot with multiple subplots.
        """
        fig = plt.figure(figsize=self.figsize)
        fig.suptitle(f'Linear Regression Analysis - R² = {r2:.4f}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Plot 1: Actual vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        self.plot_actual_vs_predicted(y_true, y_pred, r2, intercept, fitted_coef, ax1)
        
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
        Plot the effect of sample size on R² vs Adjusted R² with improved scaling.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert to numpy arrays for easier manipulation
        sample_sizes = np.array(sample_sizes)
        r2_values = np.array(r2_values)
        r2_adj_values = np.array(r2_adj_values)
        
        ax.plot(sample_sizes, r2_values, 'bo-', label='R²', linewidth=3, markersize=10)
        ax.plot(sample_sizes, r2_adj_values, 'ro-', label='Adjusted R²', linewidth=3, markersize=10)
        
        # Fill area between lines to show penalty
        ax.fill_between(sample_sizes, r2_values, r2_adj_values, alpha=0.3, color='gray', 
                       label='Adjustment Penalty')
        
        ax.set_xlabel('Sample Size (n)', fontsize=14)
        ax.set_ylabel('R² Value', fontsize=14)
        ax.set_title('R² vs Adjusted R² as Function of Sample Size\nDemonstrating Penalty Convergence', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Improve y-axis scaling to show differences better
        y_min = min(r2_adj_values.min(), r2_values.min())
        y_max = max(r2_adj_values.max(), r2_values.max())
        y_range = y_max - y_min
        
        if y_range < 0.001:  # If range is very small, expand it
            y_center = (y_min + y_max) / 2
            y_min = y_center - 0.0005
            y_max = y_center + 0.0005
        else:
            # Add 10% padding
            padding = y_range * 0.1
            y_min -= padding
            y_max += padding
        
        ax.set_ylim(y_min, y_max)
        
        # Add annotations for key points with better positioning
        differences = r2_values - r2_adj_values
        max_diff_idx = np.argmax(differences)
        max_diff = differences[max_diff_idx]
        
        if max_diff > 0.00001:  # Only annotate if difference is meaningful
            ax.annotate(f'Max Penalty: {max_diff:.5f}\n(n={sample_sizes[max_diff_idx]})',
                       xy=(sample_sizes[max_diff_idx], r2_adj_values[max_diff_idx]),
                       xytext=(sample_sizes[max_diff_idx]*3, y_min + 0.7*(y_max-y_min)),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=11, ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        # Add convergence annotation
        final_diff = differences[-1]
        ax.annotate(f'Final Difference: {final_diff:.6f}\n(n={sample_sizes[-1]})',
                   xy=(sample_sizes[-1], r2_values[-1]),
                   xytext=(sample_sizes[-1]/3, y_min + 0.3*(y_max-y_min)),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                   fontsize=11, ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        # Add formula with ASCII characters only
        formula_text = 'Adjusted R² = 1 - (1 - R²) × (n-1)/(n-p-1)'
        ax.text(0.02, 0.98, formula_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
               verticalalignment='top', fontsize=12, fontfamily='monospace')
        
        # Add data table for precise values
        table_text = f"Sample Size Analysis Results:\n"
        table_text += f"{'n':<6} {'R²':<8} {'Adj R²':<8} {'Diff':<8}\n"
        table_text += "-" * 32 + "\n"
        for i, n in enumerate(sample_sizes):
            table_text += f"{n:<6} {r2_values[i]:<8.5f} {r2_adj_values[i]:<8.5f} {differences[i]:<8.5f}\n"
        
        ax.text(0.98, 0.02, table_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
               verticalalignment='bottom', horizontalalignment='right',
               fontsize=9, fontfamily='monospace')
        
        return fig