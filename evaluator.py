"""
Model Evaluation Module for Linear Regression
Handles calculation of evaluation metrics.
"""

import numpy as np
from sklearn.metrics import r2_score
from config import setup_logging

# Setup logger
logger = setup_logging(__name__)

class RegressionEvaluator:
    """
    Handles linear regression model evaluation.
    """

    def __init__(self, trainer):
        """Initializes the evaluator.

        Args:
            trainer (LinearRegressionTrainer): The trainer containing the fitted model.
        
        Raises:
            ValueError: If the model in the trainer is not fitted.
        """
        if not trainer.is_fitted:
            logger.error("Model must be fitted before evaluation.")
            raise ValueError("Model must be fitted before evaluation.")
        self.trainer = trainer
        self.y_true = trainer.y_train
        self.y_pred = trainer.y_pred
        self.n = len(self.y_true)
        self.p = trainer.X_train.shape[1]
        logger.info("Initialized RegressionEvaluator.")

    def calculate_r2(self):
        """Calculates R² (coefficient of determination).

        Returns:
            float: R² value.
        """
        return r2_score(self.y_true, self.y_pred)

    def calculate_adjusted_r2(self):
        """Calculates adjusted R².

        Returns:
            float: Adjusted R² value.
        """
        r2 = self.calculate_r2()
        return 1 - (1 - r2) * (self.n - 1) / (self.n - self.p - 1)

    def get_residuals(self):
        """Calculates residuals (y_true - y_pred).

        Returns:
            numpy.ndarray: Residuals.
        """
        return self.y_true - self.y_pred

    def calculate_metrics(self):
        """Calculates comprehensive model evaluation metrics.

        Returns:
            dict: Dictionary containing various metrics.
        """
        logger.info("Calculating model evaluation metrics.")
        r2 = self.calculate_r2()
        r2_adj = self.calculate_adjusted_r2()
        residuals = self.get_residuals()
        
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        metrics = {
            'r2': r2,
            'adjusted_r2': r2_adj,
            'r2_difference': r2 - r2_adj,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'n_samples': self.n,
            'n_features': self.p
        }
        logger.info(f"Calculated metrics: {metrics}")
        return metrics

    def compare_with_true_parameters(self, true_intercept, true_coefficients):
        """Compares fitted parameters with true parameters.

        Args:
            true_intercept (float): True intercept value.
            true_coefficients (list): True coefficient values.

        Returns:
            dict: Comparison results.
        """
        logger.info("Comparing fitted parameters with true parameters.")
        fitted_params = self.trainer.get_coefficients()
        
        intercept_error = abs(fitted_params['intercept'] - true_intercept)
        coef_errors = [abs(fitted - true) for fitted, true in 
                      zip(fitted_params['coefficients'], true_coefficients)]
        
        comparison = {
            'true_intercept': true_intercept,
            'fitted_intercept': fitted_params['intercept'],
            'intercept_error': intercept_error,
            'true_coefficients': true_coefficients,
            'fitted_coefficients': fitted_params['coefficients'],
            'coefficient_errors': coef_errors,
            'mean_coefficient_error': np.mean(coef_errors)
        }
        logger.info(f"Parameter comparison: {comparison}")
        return comparison
