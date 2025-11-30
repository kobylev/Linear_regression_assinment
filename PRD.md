# Product Requirements Document: Linear Regression Analysis Tool

**Author:** Koby Lev
**Date:** 2025-11-30
**Version:** 1.0

## 1. Introduction

### 1.1. Purpose
This document outlines the product requirements for a Python-based tool that performs linear regression analysis on synthetic data. The primary goal is to provide a comprehensive and educational platform for understanding the nuances of linear regression, particularly the behavior of R² and Adjusted R² metrics under different conditions.

### 1.2. Scope
The project encompasses:
- Generation of synthetic datasets with configurable parameters.
- Training of linear regression models.
- Calculation and comparison of R² and Adjusted R² metrics.
- Analysis of the impact of sample size and predictor count on model evaluation.
- Comprehensive visualization of results for diagnostic and educational purposes.

### 1.3. Intended Audience
This tool is designed for:
- **Data science students and educators:** To learn and teach the principles of linear regression.
- **Researchers and analysts:** To simulate data and understand model behavior before applying techniques to real-world problems.
- **Python developers:** As a reference for best practices in structuring a data analysis project.

## 2. Functional Requirements

### 2.1. Data Generation
- **FR1.1:** The system must generate a synthetic dataset (X, y) based on the linear equation y = β₀ + β₁x₁ + ... + βₚxₚ + ε.
- **FR1.2:** The user must be able to configure the following parameters for data generation:
    - Number of samples (
).
    - Number of predictors (p).
    - Standard deviation of the noise term (ε).
    - True intercept (β₀) and coefficients (β₁ to βₚ).
- **FR1.3:** The generated features (predictors) should be drawn from normal distributions with configurable means and standard deviations.

### 2.2. Model Training
- **FR2.1:** The system must train a linear regression model on the generated dataset using the Ordinary Least Squares (OLS) method.
- **FR2.2:** The system must extract the fitted intercept and coefficients from the trained model.

### 2.3. Performance Evaluation
- **FR3.1:** The system must calculate the **R² (Coefficient of Determination)** for the trained model.
- **FR3.2:** The system must calculate the **Adjusted R²** using the formula: R²ₐdⱼ = 1 - (1 - R²) * (n-1)/(n-p-1).
- **FR3.3:** The system must calculate other standard regression metrics, including:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
- **FR3.4:** The system must provide a comparison between the true model parameters and the fitted parameters.

### 2.4. Analysis and Visualization
- **FR4.1:** The system must generate a multi-panel diagnostic plot that includes:
    - Actual vs. Predicted values.
    - Residuals vs. Predicted values.
    - A histogram of residuals with a normal distribution overlay.
    - A Q-Q plot of residuals.
    - A heatmap of feature correlations.
    - A bar chart comparing true vs. fitted coefficients.
- **FR4.2:** The system must perform an analysis on the effect of varying sample sizes on R² and Adjusted R², and visualize the convergence of the two metrics.
- **FR4.3:** All plots must be saved as high-resolution PNG files.

### 2.5. Main Execution Script
- **FR5.1:** A main script (main.py) must orchestrate the entire workflow: data generation, model training, evaluation, and visualization.
- **FR5.2:** The main script must output a formatted summary of the results to the console.

## 3. Non-Functional Requirements

### 3.1. Performance
- **NFR1.1:** The tool should complete a full analysis run (e.g., n=1000, p=5) in under 30 seconds on a standard machine.
- **NFR1.2:** The sample size analysis, which involves multiple runs, should complete in a reasonable timeframe (e.g., under 2 minutes).

### 3.2. Usability
- **NFR2.1:** The code must be well-documented with clear comments and docstrings.
- **NFR2.2:** The project structure must be modular and easy to navigate.
- **NFR2.3:** A README.md file must provide clear instructions on setup, usage, and interpretation of results.

### 3.3. Technical Stack
- **NFR3.1:** The project must be implemented in Python (version 3.7+).
- **NFR3.2:** The project will use the following core libraries:
    - 
umpy for numerical operations.
    - pandas for data manipulation.
- **NFR3.3:** The project will use the following libraries for modeling and visualization:
    - scikit-learn for linear regression modeling.
    - matplotlib and seaborn for plotting.
    - scipy for statistical calculations (e.g., Q-Q plot).

### 3.4. Reproducibility
- **NFR4.1:** The data generation process must use a fixed random seed to ensure that results are reproducible.

## 4. Success Criteria

The project will be considered successful when the following criteria are met:
- **SC1:** All functional requirements listed in Section 2 are implemented and working correctly.
- **SC2:** The generated visualizations and console outputs clearly demonstrate the key theoretical concepts, such as:
    - The model's ability to recover the true parameters from the synthetic data.
    - The relationship Adjusted R² ≤ R².
    - The convergence of R² and Adjusted R² as the sample size increases.
- **SC3:** The README.md file is comprehensive and provides a clear, educational walkthrough of the project and its findings.
- **SC4:** The codebase adheres to good programming practices, including modularity, documentation, and readability.

## 5. Future Work (Out of Scope for V1.0)
- **FW1:** Implementation of other regression techniques (e.g., Ridge, Lasso).
- **FW2:** Adding cross-validation to the evaluation workflow.
- **FW3:** Analysis of the impact of multicollinearity.
- **FW4:** Building an interactive user interface (e.g., with Streamlit or Dash) to configure and run analyses.
