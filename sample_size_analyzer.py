"""
Sample Size Analysis Module
Analyzes the effect of different sample sizes on R² and Adjusted R².
"""

import numpy as np
from data_generator import SyntheticDataGenerator
from model_trainer import LinearRegressionTrainer

class SampleSizeAnalyzer:
    """
    Analyzes the relationship between sample size and model performance metrics.
    """
    
    def __init__(self, base_generator):
        """
        Initialize the analyzer with a base data generator.
        
        Parameters:
        -----------
        base_generator : SyntheticDataGenerator
            Base data generator to use for analysis
        """
        self.base_generator = base_generator
    
    def analyze_sample_sizes(self, sample_sizes=None):
        """
        Analyze the effect of different sample sizes on R² and Adjusted R².
        
        Parameters:
        -----------
        sample_sizes : list, optional
            List of sample sizes to test
        
        Returns:
        --------
        dict
            Analysis results containing metrics for each sample size
        """
        if sample_sizes is None:
            sample_sizes = [50, 100, 500, 1000, 2000, 5000]
        
        results = {
            'sample_sizes': sample_sizes,
            'r2_values': [],
            'r2_adj_values': [],
            'differences': [],
            'detailed_results': []
        }
        
        for n in sample_sizes:
            # Create generator with specific sample size
            generator = SyntheticDataGenerator(
                n=n,
                p=self.base_generator.p,
                noise_std=self.base_generator.noise_std,
                random_seed=self.base_generator.random_seed
            )
            generator.set_true_parameters(
                self.base_generator.beta_0,
                self.base_generator.beta_coefficients
            )
            
            # Generate data
            X, y, data_info = generator.generate_dataset()
            
            # Train model
            trainer = LinearRegressionTrainer()
            trainer.fit(X, y)
            
            # Calculate metrics
            metrics = trainer.calculate_metrics()
            
            # Store results
            results['r2_values'].append(metrics['r2'])
            results['r2_adj_values'].append(metrics['adjusted_r2'])
            results['differences'].append(metrics['r2_difference'])
            
            # Store detailed results
            detailed = {
                'n': n,
                'p': self.base_generator.p,
                'r2': metrics['r2'],
                'adjusted_r2': metrics['adjusted_r2'],
                'difference': metrics['r2_difference'],
                'mse': metrics['mse'],
                'rmse': metrics['rmse']
            }
            results['detailed_results'].append(detailed)
        
        return results
    
    def print_sample_size_analysis(self, results):
        """
        Print formatted sample size analysis results.
        
        Parameters:
        -----------
        results : dict
            Results from analyze_sample_sizes method
        """
        print("\n=== SAMPLE SIZE ANALYSIS ===")
        print(f"{'Sample Size':>12} {'R²':>8} {'Adj R²':>8} {'Difference':>10} {'RMSE':>8}")
        print("-" * 55)
        
        for detail in results['detailed_results']:
            print(f"{detail['n']:>12} {detail['r2']:>8.4f} {detail['adjusted_r2']:>8.4f} "
                  f"{detail['difference']:>10.4f} {detail['rmse']:>8.4f}")
        
        print(f"\nKey Observations:")
        print(f"- As sample size increases, R² and Adjusted R² converge")
        print(f"- Largest difference: {max(results['differences']):.4f} (smallest sample)")
        print(f"- Smallest difference: {min(results['differences']):.4f} (largest sample)")
        print(f"- Penalty decreases as n/{results['detailed_results'][0]['p']} ratio increases")
    
    def analyze_predictor_impact(self, predictor_counts=None, n=1000):
        """
        Analyze the impact of number of predictors on R² vs Adjusted R².
        
        Parameters:
        -----------
        predictor_counts : list, optional
            List of predictor counts to test
        n : int
            Sample size to use for all tests
        
        Returns:
        --------
        dict
            Analysis results for different predictor counts
        """
        if predictor_counts is None:
            predictor_counts = [2, 4, 6, 8, 10, 15]
        
        results = {
            'predictor_counts': predictor_counts,
            'r2_values': [],
            'r2_adj_values': [],
            'differences': [],
            'detailed_results': []
        }
        
        for p in predictor_counts:
            # Create coefficients for p predictors
            base_coef = [2.5, -1.8, 3.2, 0.7, 1.2, -0.9, 2.1, -1.5, 0.8, -2.3, 1.7, -1.1, 2.8, -0.6, 1.9]
            coefficients = base_coef[:p]
            
            # Create generator
            generator = SyntheticDataGenerator(
                n=n,
                p=p,
                noise_std=self.base_generator.noise_std,
                random_seed=self.base_generator.random_seed
            )
            generator.set_true_parameters(self.base_generator.beta_0, coefficients)
            
            # Generate data
            X, y, data_info = generator.generate_dataset()
            
            # Train model
            trainer = LinearRegressionTrainer()
            trainer.fit(X, y)
            
            # Calculate metrics
            metrics = trainer.calculate_metrics()
            
            # Store results
            results['r2_values'].append(metrics['r2'])
            results['r2_adj_values'].append(metrics['adjusted_r2'])
            results['differences'].append(metrics['r2_difference'])
            
            # Store detailed results
            detailed = {
                'p': p,
                'n': n,
                'r2': metrics['r2'],
                'adjusted_r2': metrics['adjusted_r2'],
                'difference': metrics['r2_difference'],
                'penalty_factor': (n - 1) / (n - p - 1)
            }
            results['detailed_results'].append(detailed)
        
        return results
    
    def print_predictor_analysis(self, results):
        """
        Print formatted predictor impact analysis results.
        
        Parameters:
        -----------
        results : dict
            Results from analyze_predictor_impact method
        """
        print("\n=== PREDICTOR COUNT ANALYSIS ===")
        print(f"{'Predictors':>11} {'R²':>8} {'Adj R²':>8} {'Difference':>10} {'Penalty Factor':>14}")
        print("-" * 65)
        
        for detail in results['detailed_results']:
            print(f"{detail['p']:>11} {detail['r2']:>8.4f} {detail['adjusted_r2']:>8.4f} "
                  f"{detail['difference']:>10.4f} {detail['penalty_factor']:>14.4f}")
        
        print(f"\nKey Observations:")
        print(f"- More predictors increase the penalty in Adjusted R²")
        print(f"- Penalty factor = (n-1)/(n-p-1) increases with more predictors")
        print(f"- Largest difference: {max(results['differences']):.4f} (most predictors)")
        print(f"- With n={results['detailed_results'][0]['n']}, penalty becomes significant with many predictors")