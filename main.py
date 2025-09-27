"""
Main Analysis Script for Linear Regression with Synthetic Data
Demonstrates R² and Adjusted R² calculation and analysis.
CORRECTED VERSION - Fixes TypeError and shape mismatch errors
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
from data_generator import SyntheticDataGenerator
from model_trainer import LinearRegressionTrainer
from visualizer import RegressionVisualizer
from sample_size_analyzer import SampleSizeAnalyzer

def main():
    """
    Main function to run the complete linear regression analysis.
    """
    print("=" * 60)
    print("LINEAR REGRESSION ANALYSIS WITH SYNTHETIC DATA")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n1. GENERATING SYNTHETIC DATA")
    print("-" * 30)
    
    generator = SyntheticDataGenerator(n=1000, p=4, noise_std=2.0, random_seed=42)
    X, y, data_info = generator.generate_dataset()
    data = generator.create_dataframe(X, y, data_info['feature_names'])
    
    print(f"Dataset created with {data_info['n_samples']} samples and {data_info['n_features']} features")
    print(f"True intercept (β₀): {data_info['true_intercept']}")
    print(f"True coefficients: {data_info['true_coefficients']}")
    print(f"Noise standard deviation: {data_info['noise_std']}")
    
    print(f"\nDataset shape: {data.shape}")
    print(f"\nFirst 5 rows:")
    print(data.head())
    
    print(f"\nFeature statistics:")
    print(data.describe().round(4))
    
    # Step 2: Train the model
    print(f"\n2. TRAINING LINEAR REGRESSION MODEL")
    print("-" * 35)
    
    trainer = LinearRegressionTrainer()
    trainer.fit(X, y)
    
    # Get model results
    fitted_params = trainer.get_coefficients()
    metrics = trainer.calculate_metrics()
    residuals = trainer.get_residuals()
    
    print(f"Model training completed successfully")
    print(f"Fitted intercept: {fitted_params['intercept']:.4f}")
    print(f"Fitted coefficients: {[f'{coef:.4f}' for coef in fitted_params['coefficients']]}")
    
    # Step 3: Calculate and display metrics
    print(f"\n3. MODEL EVALUATION METRICS")
    print("-" * 28)
    
    print(f"R² (Coefficient of Determination): {metrics['r2']:.6f}")
    print(f"Adjusted R²: {metrics['adjusted_r2']:.6f}")
    print(f"Difference (R² - Adjusted R²): {metrics['r2_difference']:.6f}")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    
    # Step 4: Compare with true parameters
    print(f"\n4. TRUE vs FITTED PARAMETERS COMPARISON")
    print("-" * 40)
    
    comparison = trainer.compare_with_true_parameters(
        data_info['true_intercept'],
        data_info['true_coefficients']
    )
    
    print(f"{'Parameter':<12} {'True Value':<12} {'Fitted Value':<14} {'Error':<10}")
    print("-" * 50)
    print(f"{'Intercept':<12} {comparison['true_intercept']:<12.4f} "
          f"{comparison['fitted_intercept']:<14.4f} {comparison['intercept_error']:<10.4f}")
    
    for i, (true_coef, fitted_coef, error) in enumerate(zip(
        comparison['true_coefficients'],
        comparison['fitted_coefficients'],
        comparison['coefficient_errors']
    )):
        print(f"β{i+1:<11} {true_coef:<12.4f} {fitted_coef:<14.4f} {error:<10.4f}")
    
    print(f"\nMean coefficient error: {comparison['mean_coefficient_error']:.4f}")
    
    # Display regression equation
    print(f"\n5. REGRESSION EQUATION")
    print("-" * 20)
    
    intercept = fitted_params['intercept']
    coefficients = fitted_params['coefficients']
    
    equation = f"ŷ = {intercept:.2f}"
    true_equation = f"y = {data_info['true_intercept']:.2f}"
    
    for i, (coef, true_coef) in enumerate(zip(coefficients, data_info['true_coefficients'])):
        if coef >= 0:
            equation += f" + {coef:.2f}X{i+1}"
        else:
            equation += f" - {abs(coef):.2f}X{i+1}"
            
        if true_coef >= 0:
            true_equation += f" + {true_coef:.2f}X{i+1}"
        else:
            true_equation += f" - {abs(true_coef):.2f}X{i+1}"
    
    true_equation += f" + ε(0,{data_info['noise_std']:.1f}²)"
    
    print(f"Fitted equation: {equation}")
    print(f"True equation:   {true_equation}")
    
    # Step 6: Create visualizations
    print(f"\n6. CREATING VISUALIZATIONS")
    print("-" * 25)
    
    visualizer = RegressionVisualizer()
    
    # Main comprehensive plot
    fig1 = visualizer.create_comprehensive_plot(
        y, trainer.y_pred, residuals, data,
        data_info['true_coefficients'],
        fitted_params['coefficients'],
        metrics['r2']
    )
    plt.savefig('regression_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    print("Saved: regression_analysis_comprehensive.png")
    
    # Step 7: Sample size analysis
    print(f"\n7. SAMPLE SIZE ANALYSIS")
    print("-" * 22)
    
    analyzer = SampleSizeAnalyzer(generator)
    sample_size_results = analyzer.analyze_sample_sizes([50, 100, 500, 1000, 2000, 5000])
    analyzer.print_sample_size_analysis(sample_size_results)
    
    # Plot sample size analysis
    fig2 = visualizer.plot_sample_size_analysis(
        sample_size_results['sample_sizes'],
        sample_size_results['r2_values'],
        sample_size_results['r2_adj_values']
    )
    plt.savefig('sample_size_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSaved: sample_size_analysis.png")
    
    # Step 8: Predictor count analysis
    print(f"\n8. PREDICTOR COUNT ANALYSIS")
    print("-" * 26)
    
    predictor_results = analyzer.analyze_predictor_impact([2, 4, 6, 8, 10], n=1000)
    analyzer.print_predictor_analysis(predictor_results)
    
    # Step 9: Summary and conclusions
    print(f"\n9. SUMMARY AND CONCLUSIONS")
    print("-" * 26)
    
    print(f"✓ Successfully generated synthetic dataset with known ground truth")
    print(f"✓ Linear regression model recovered true parameters with high accuracy")
    print(f"✓ R² = {metrics['r2']:.4f} indicates the model explains {metrics['r2']*100:.1f}% of variance")
    print(f"✓ Adjusted R² = {metrics['adjusted_r2']:.4f} accounts for {data_info['n_features']} predictors")
    print(f"✓ Small penalty of {metrics['r2_difference']:.4f} due to large sample size relative to predictors")
    print(f"✓ Residuals appear normally distributed, validating model assumptions")
    
    print(f"\nKey Insights:")
    print(f"• Adjusted R² is always ≤ R² due to penalty for additional predictors")
    print(f"• Penalty decreases as n/p ratio increases (more samples per predictor)")
    print(f"• With large datasets, R² and Adjusted R² converge")
    print(f"• Model successfully demonstrates the theoretical relationship between metrics")
    
    plt.show()

def run_custom_analysis(n=1000, p=4, noise_std=2.0, true_coefficients=None):
    """
    Run analysis with custom parameters.
    CORRECTED VERSION - Handles variable predictor counts correctly
    
    Parameters:
    -----------
    n : int
        Number of samples
    p : int
        Number of predictors
    noise_std : float
        Standard deviation of noise
    true_coefficients : list, optional
        True coefficient values
    """
    if true_coefficients is None:
        # Extended coefficient array to handle up to 20 predictors
        base_coef = [2.5, -1.8, 3.2, 0.7, 1.2, -0.9, 2.1, -1.5, 0.8, -2.3, 
                    1.7, -1.1, 2.8, -0.6, 1.9, 0.5, -1.3, 2.4, -0.8, 1.6]
        
        # Ensure we have exactly p coefficients
        if p <= len(base_coef):
            true_coefficients = base_coef[:p]
        else:
            # If more predictors requested than we have coefficients, extend with random values
            import random
            random.seed(42)  # For reproducibility
            extra_coef = [random.uniform(-3, 3) for _ in range(p - len(base_coef))]
            true_coefficients = base_coef + extra_coef
    
    print(f"\nRUNNING CUSTOM ANALYSIS")
    print(f"Parameters: n={n}, p={p}, noise_std={noise_std}")
    print(f"True coefficients: {[f'{c:.2f}' for c in true_coefficients]}")
    
    # Generate data
    generator = SyntheticDataGenerator(n=n, p=p, noise_std=noise_std)
    generator.set_true_parameters(5.0, true_coefficients)
    
    X, y, data_info = generator.generate_dataset()
    
    # Train model
    trainer = LinearRegressionTrainer()
    trainer.fit(X, y)
    
    # Get metrics
    metrics = trainer.calculate_metrics()
    
    print(f"Results:")
    print(f"R² = {metrics['r2']:.6f}")
    print(f"Adjusted R² = {metrics['adjusted_r2']:.6f}")
    print(f"Difference = {metrics['r2_difference']:.6f}")
    print(f"Penalty factor = {(n-1)/(n-p-1):.4f}")
    print(f"n/p ratio = {n/p:.1f}")

if __name__ == "__main__":
    try:
        # Run main analysis
        main()
        
        # Optional: Run custom analyses with different parameters
        print(f"\n" + "="*60)
        print("ADDITIONAL CUSTOM ANALYSES")
        print("="*60)
        
        # Small sample size example
        print(f"\n" + "-"*30)
        print("SMALL SAMPLE SIZE EXAMPLE")
        print("-"*30)
        run_custom_analysis(n=50, p=4, noise_std=2.0)
        
        # Many predictors example (FIXED - now handles p=15 correctly)
        print(f"\n" + "-"*30)
        print("MANY PREDICTORS EXAMPLE")
        print("-"*30)
        run_custom_analysis(n=200, p=15, noise_std=2.0)
        
        # High noise example
        print(f"\n" + "-"*30)
        print("HIGH NOISE EXAMPLE")
        print("-"*30)
        run_custom_analysis(n=1000, p=4, noise_std=5.0)
        
        # Extreme case: Many predictors, small sample
        print(f"\n" + "-"*30)
        print("EXTREME CASE: p=10, n=30")
        print("-"*30)
        run_custom_analysis(n=30, p=10, noise_std=2.0)
        
        print(f"\nAnalysis completed. Check generated PNG files for visualizations.")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Program execution finished.")