"""
Predictor Impact Analysis Module
Analyzes the effect of the number of predictors on R² and Adjusted R².
"""

from data_generator import SyntheticDataGenerator
from model_trainer import LinearRegressionTrainer
from config import setup_logging

# Setup logger
logger = setup_logging(__name__)

class PredictorImpactAnalyzer:
    """
    Analyzes the relationship between number of predictors and model performance metrics.
    """
    
    def __init__(self, base_generator):
        """Initializes the analyzer with a base data generator.

        Args:
            base_generator (SyntheticDataGenerator): Base data generator to use for analysis.
        """
        self.base_generator = base_generator
        logger.info("Initialized PredictorImpactAnalyzer.")

    def analyze_predictor_impact(self, predictor_counts=None, n=1000):
        """Analyzes the impact of number of predictors on R² vs Adjusted R².

        Args:
            predictor_counts (list, optional): List of predictor counts to test. Defaults to None.
            n (int, optional): Sample size to use for all tests. Defaults to 1000.

        Returns:
            dict: Analysis results for different predictor counts.
        """
        if predictor_counts is None:
            predictor_counts = [2, 4, 6, 8, 10, 15]
        
        logger.info(f"Analyzing predictor impact with counts: {predictor_counts} and n={n}")

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
            X, y, _ = generator.generate_dataset()
            
            # Train model
            trainer = LinearRegressionTrainer()
            evaluator = trainer.fit(X, y)
            
            # Calculate metrics
            metrics = evaluator.calculate_metrics()
            
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
        
        logger.info("Predictor impact analysis completed.")
        return results
    
    def print_predictor_analysis(self, results):
        """Prints formatted predictor impact analysis results.

        Args:
            results (dict): Results from analyze_predictor_impact method.
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
