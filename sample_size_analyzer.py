"""
Sample Size Analysis Module
Analyzes the effect of different sample sizes on R² and Adjusted R².
"""

from data_generator import SyntheticDataGenerator
from model_trainer import LinearRegressionTrainer
from config import setup_logging

# Setup logger
logger = setup_logging(__name__)

class SampleSizeAnalyzer:
    """
    Analyzes the relationship between sample size and model performance metrics.
    """
    
    def __init__(self, base_generator):
        """Initializes the analyzer with a base data generator.

        Args:
            base_generator (SyntheticDataGenerator): Base data generator to use for analysis.
        """
        self.base_generator = base_generator
        logger.info("Initialized SampleSizeAnalyzer.")
    
    def analyze_sample_sizes(self, sample_sizes=None):
        """Analyzes the effect of different sample sizes on R² and Adjusted R².

        Args:
            sample_sizes (list, optional): List of sample sizes to test. Defaults to None.

        Returns:
            dict: Analysis results containing metrics for each sample size.
        """
        if sample_sizes is None:
            sample_sizes = [50, 100, 500, 1000, 2000, 5000]
        
        logger.info(f"Analyzing sample sizes: {sample_sizes}")

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
                'n': n,
                'p': self.base_generator.p,
                'r2': metrics['r2'],
                'adjusted_r2': metrics['adjusted_r2'],
                'difference': metrics['r2_difference'],
                'mse': metrics['mse'],
                'rmse': metrics['rmse']
            }
            results['detailed_results'].append(detailed)
        
        logger.info("Sample size analysis completed.")
        return results
    
    def print_sample_size_analysis(self, results):
        """Prints formatted sample size analysis results.

        Args:
            results (dict): Results from analyze_sample_sizes method.
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