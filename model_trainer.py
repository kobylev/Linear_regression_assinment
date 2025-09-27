"""
Model Training Module for Linear Regression
Handles model fitting and evaluation metrics calculation.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class LinearRegressionTrainer:
    """
    Handles linear regression model training and evaluation.
    """
    
    def __init__(self):
        """
        Initialize the trainer.
        """
        self.model = None
        self.is_fitted = False
        self.X_train = None
        self.y_train = None
        self.y_pred = None
    
    def fit(self, X, y):
        """
        Fit the linear regression model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Response variable
        """
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.is_fitted = True
        self.X_train = X
        self.y_train = y
        self.y_pred = self.model.predict(X)
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix for prediction
        
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def calculate_r2(self):
        """
        Calculate R² (coefficient of determination).
        
        Returns:
        --------
        float
            R² value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating R²")
        
        return r2_score(self.y_train, self.y_pred)
    
    def calculate_adjusted_r2(self, n=None, p=None):
        """
        Calculate adjusted R² using the formula:
        R²_adj = 1 - (1 - R²) * (n - 1) / (n - p - 1)
        
        Parameters:
        -----------
        n : int, optional
            Number of observations (if None, inferred from training data)
        p : int, optional
            Number of predictors (if None, inferred from training data)
        
        Returns:
        --------
        float
            Adjusted R² value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating adjusted R²")
        
        # Infer n and p if not provided
        if n is None:
            n = len(self.y_train)
        if p is None:
            p = self.X_train.shape[1]
        
        r2 = self.calculate_r2()
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        return r2_adj
    
    def get_coefficients(self):
        """
        Get the fitted model coefficients.
        
        Returns:
        --------
        dict
            Dictionary containing intercept and coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting coefficients")
        
        return {
            'intercept': self.model.intercept_,
            'coefficients': self.model.coef_.tolist()
        }
    
    def get_residuals(self):
        """
        Calculate residuals (y_true - y_pred).
        
        Returns:
        --------
        numpy.ndarray
            Residuals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating residuals")
        
        return self.y_train - self.y_pred
    
    def calculate_metrics(self):
        """
        Calculate comprehensive model evaluation metrics.
        
        Returns:
        --------
        dict
            Dictionary containing various metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating metrics")
        
        n = len(self.y_train)
        p = self.X_train.shape[1]
        
        r2 = self.calculate_r2()
        r2_adj = self.calculate_adjusted_r2(n, p)
        residuals = self.get_residuals()
        
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        return {
            'r2': r2,
            'adjusted_r2': r2_adj,
            'r2_difference': r2 - r2_adj,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'n_samples': n,
            'n_features': p
        }
    
    def compare_with_true_parameters(self, true_intercept, true_coefficients):
        """
        Compare fitted parameters with true parameters.
        
        Parameters:
        -----------
        true_intercept : float
            True intercept value
        true_coefficients : list
            True coefficient values
        
        Returns:
        --------
        dict
            Comparison results
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before comparing parameters")
        
        fitted_params = self.get_coefficients()
        
        intercept_error = abs(fitted_params['intercept'] - true_intercept)
        coef_errors = [abs(fitted - true) for fitted, true in 
                      zip(fitted_params['coefficients'], true_coefficients)]
        
        return {
            'true_intercept': true_intercept,
            'fitted_intercept': fitted_params['intercept'],
            'intercept_error': intercept_error,
            'true_coefficients': true_coefficients,
            'fitted_coefficients': fitted_params['coefficients'],
            'coefficient_errors': coef_errors,
            'mean_coefficient_error': np.mean(coef_errors)
        }