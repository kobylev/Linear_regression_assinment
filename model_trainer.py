"""
Model Training Module for Linear Regression
Handles model fitting.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from config import setup_logging
from evaluator import RegressionEvaluator

# Setup logger
logger = setup_logging(__name__)

class LinearRegressionTrainer:
    """
    Handles linear regression model training.
    """
    
    def __init__(self):
        """Initializes the trainer."""
        self.model = None
        self.is_fitted = False
        self.X_train = None
        self.y_train = None
        self.y_pred = None
        logger.info("Initialized LinearRegressionTrainer.")
    
    def fit(self, X, y):
        """Fits the linear regression model.

        Args:
            X (numpy.ndarray): Feature matrix.
            y (numpy.ndarray): Response variable.

        Returns:
            RegressionEvaluator: An evaluator instance for the fitted model.
        """
        logger.info(f"Fitting model with {X.shape[0]} samples and {X.shape[1]} features.")
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.is_fitted = True
        self.X_train = X
        self.y_train = y
        self.y_pred = self.model.predict(X)
        logger.info("Model fitting completed.")
        return RegressionEvaluator(self)
    
    def predict(self, X):
        """Makes predictions using the fitted model.

        Args:
            X (numpy.ndarray): Feature matrix for prediction.

        Returns:
            numpy.ndarray: Predicted values.
        
        Raises:
            ValueError: If the model is not fitted before making predictions.
        """
        if not self.is_fitted:
            logger.error("Model must be fitted before making predictions")
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_coefficients(self):
        """Gets the fitted model coefficients.

        Returns:
            dict: Dictionary containing intercept and coefficients.

        Raises:
            ValueError: If the model is not fitted before getting coefficients.
        """
        if not self.is_fitted:
            logger.error("Model must be fitted before getting coefficients")
            raise ValueError("Model must be fitted before getting coefficients")
        
        return {
            'intercept': self.model.intercept_,
            'coefficients': self.model.coef_.tolist()
        }