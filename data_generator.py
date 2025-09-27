"""
Data Generator Module for Linear Regression Synthetic Dataset
Handles the generation of synthetic data following a linear relationship.
"""

import numpy as np
import pandas as pd

class SyntheticDataGenerator:
    """
    Generates synthetic data for linear regression problems.
    """
    
    def __init__(self, n=1000, p=4, noise_std=2.0, random_seed=42):
        """
        Initialize the data generator.
        
        Parameters:
        -----------
        n : int
            Number of observations/samples
        p : int
            Number of predictors/features
        noise_std : float
            Standard deviation of the noise term
        random_seed : int
            Random seed for reproducibility
        """
        self.n = n
        self.p = p
        self.noise_std = noise_std
        self.random_seed = random_seed
        
        # Set default true parameters
        self.beta_0 = 5.0  # intercept
        self.beta_coefficients = [2.5, -1.8, 3.2, 0.7][:p]  # coefficients
        
        # Set random seed
        np.random.seed(random_seed)
    
    def set_true_parameters(self, beta_0, beta_coefficients):
        """
        Set the true parameters for the linear model.
        
        Parameters:
        -----------
        beta_0 : float
            True intercept
        beta_coefficients : list
            True coefficients for predictors
        """
        self.beta_0 = beta_0
        self.beta_coefficients = beta_coefficients
    
    def generate_features(self):
        """
        Generate input features X₁, X₂, ..., Xₚ.
        Each feature follows a normal distribution with different parameters.
        
        Returns:
        --------
        numpy.ndarray
            Feature matrix of shape (n, p)
        """
        # Generate features with different distributions
        features = []
        
        # Predefined parameters for different features
        feature_params = [
            (25, 10),  # X1: mean=25, std=10
            (30, 8),   # X2: mean=30, std=8
            (20, 12),  # X3: mean=20, std=12
            (15, 6),   # X4: mean=15, std=6
            (35, 9),   # X5: mean=35, std=9
            (40, 7),   # X6: mean=40, std=7
        ]
        
        for i in range(self.p):
            if i < len(feature_params):
                mean, std = feature_params[i]
            else:
                # Default parameters for additional features
                mean, std = 25, 10
            
            feature = np.random.normal(mean, std, self.n)
            features.append(feature)
        
        return np.column_stack(features)
    
    def generate_response(self, X):
        """
        Generate response variable y using the linear model:
        y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        
        Returns:
        --------
        numpy.ndarray
            Response variable
        """
        # Generate noise term
        epsilon = np.random.normal(0, self.noise_std, self.n)
        
        # Calculate linear combination
        linear_combination = self.beta_0 + np.dot(X, self.beta_coefficients)
        
        # Add noise
        y = linear_combination + epsilon
        
        return y
    
    def generate_dataset(self):
        """
        Generate complete synthetic dataset.
        
        Returns:
        --------
        tuple
            (X, y, data_info) where:
            - X: feature matrix
            - y: response variable
            - data_info: dictionary with dataset information
        """
        # Generate features
        X = self.generate_features()
        
        # Generate response
        y = self.generate_response(X)
        
        # Create info dictionary
        data_info = {
            'n_samples': self.n,
            'n_features': self.p,
            'true_intercept': self.beta_0,
            'true_coefficients': self.beta_coefficients.copy(),
            'noise_std': self.noise_std,
            'feature_names': [f'X{i+1}' for i in range(self.p)]
        }
        
        return X, y, data_info
    
    def create_dataframe(self, X, y, feature_names=None):
        """
        Create a pandas DataFrame from the generated data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Response variable
        feature_names : list, optional
            Names for features
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing all variables
        """
        if feature_names is None:
            feature_names = [f'X{i+1}' for i in range(X.shape[1])]
        
        # Create DataFrame
        data = pd.DataFrame(X, columns=feature_names)
        data['y'] = y
        
        return data