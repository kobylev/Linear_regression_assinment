"""
Main Analysis Script for Linear Regression with Synthetic Data
Demonstrates R² and Adjusted R² calculation and analysis.
"""

from data_generator import SyntheticDataGenerator
from model_trainer import LinearRegressionTrainer
from sample_size_analyzer import SampleSizeAnalyzer
from predictor_analyzer import PredictorImpactAnalyzer
from visualizer import Visualizer
from config import setup_logging

# Setup logger
logger = setup_logging(__name__)

def generate_data(n, p, noise_std, random_seed):
    """Generates synthetic data."""
    logger.info("Generating synthetic data...")
    generator = SyntheticDataGenerator(n=n, p=p, noise_std=noise_std, random_seed=random_seed)
    X, y, data_info = generator.generate_dataset()
    data = generator.create_dataframe(X, y, data_info['feature_names'])
    logger.info("Data generation complete.")
    return generator, X, y, data, data_info

def train_and_evaluate_model(X, y, data_info):
    """Trains the model and evaluates its performance."""
    logger.info("Training and evaluating model...")
    trainer = LinearRegressionTrainer()
    evaluator = trainer.fit(X, y)
    metrics = evaluator.calculate_metrics()
    comparison = evaluator.compare_with_true_parameters(
        data_info['true_intercept'], data_info['true_coefficients']
    )
    logger.info("Model training and evaluation complete.")
    return trainer, evaluator, metrics, comparison

def run_and_print_sample_size_analysis(generator):
    """Runs and prints the sample size analysis."""
    logger.info("Running sample size analysis...")
    analyzer = SampleSizeAnalyzer(generator)
    results = analyzer.analyze_sample_sizes([50, 100, 500, 1000, 2000, 5000])
    analyzer.print_sample_size_analysis(results)
    logger.info("Sample size analysis complete.")
    return results

def run_and_print_predictor_impact_analysis(generator):
    """Runs and prints the predictor impact analysis."""
    logger.info("Running predictor impact analysis...")
    analyzer = PredictorImpactAnalyzer(generator)
    results = analyzer.analyze_predictor_impact([2, 4, 6, 8, 10], n=1000)
    analyzer.print_predictor_analysis(results)
    logger.info("Predictor impact analysis complete.")
    return results

def create_visualizations(visualizer, y, trainer, evaluator, data, data_info, metrics, sample_size_results):
    """Creates and saves the visualizations."""
    logger.info("Creating visualizations...")
    visualizer.create_comprehensive_plot(
        y, trainer.y_pred, evaluator.get_residuals(), data,
        data_info['true_coefficients'],
        trainer.get_coefficients()['coefficients'],
        metrics['r2'],
        trainer.get_coefficients()['intercept']
    )
    visualizer.plot_sample_size_analysis(
        sample_size_results['sample_sizes'],
        sample_size_results['r2_values'],
        sample_size_results['r2_adj_values']
    )
    logger.info("Visualizations created.")

def main():
    """
    Main function to run the complete linear regression analysis.
    """
    logger.info("Starting linear regression analysis.")
    
    # Configuration
    N_SAMPLES = 1000
    N_FEATURES = 4
    NOISE_STD = 2.0
    RANDOM_SEED = 42

    # Workflow
    generator, X, y, data, data_info = generate_data(N_SAMPLES, N_FEATURES, NOISE_STD, RANDOM_SEED)
    trainer, evaluator, metrics, _ = train_and_evaluate_model(X, y, data_info)
    sample_size_results = run_and_print_sample_size_analysis(generator)
    run_and_print_predictor_impact_analysis(generator)
    
    visualizer = Visualizer()
    create_visualizations(visualizer, y, trainer, evaluator, data, data_info, metrics, sample_size_results)

    logger.info("Linear regression analysis finished.")

if __name__ == "__main__":
    main()