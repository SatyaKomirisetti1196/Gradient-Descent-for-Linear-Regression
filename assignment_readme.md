# Linear Regression Implementation - assignment.py

**Student:** Satya Varalakshmi Komirisetti
 **Student ID:** 700773849
**Course:** CS5710 Machine Learning  
**Assignment:** Homework 1 - Linear Regression Comparison

## Overview

This Python script implements and compares two methods for linear regression:
1. **Closed-form solution** using Normal Equation
2. **Gradient Descent** implemented from scratch

## Implementation Structure

### 1. Data Generation
```python
y = 3 + 4x + noise  # 200 samples with Gaussian noise
```

### 2. Closed-Form Solution
- Uses Normal Equation: `θ = (X^T X)^(-1) X^T y`
- Direct matrix computation for optimal parameters

### 3. Gradient Descent
- **Learning rate:** 0.05
- **Iterations:** 1000
- **Loss function:** Mean Squared Error (MSE)
- Updates: `θ = θ - η * gradients`

### 4. Visualization
- Scatter plot of synthetic data
- Regression lines from both methods
- MSE loss convergence curve

## Key Features

- **Reproducible results** with `np.random.seed(42)`
- **Design matrix** with bias column for intercept term
- **Parameter comparison** between methods
- **Loss tracking** during gradient descent

## Expected Output

- Two side-by-side plots showing data fit and loss curve
- Parameter values from both methods (should be nearly identical)
- Comparison with true parameters (3, 4)
- Explanation of convergence behavior

- **Plot 1:** Raw data points with both regression lines:
  - Blue dots: synthetic data
  - Red line: closed-form regression line
  - Green dashed line: gradient descent regression line
- **Plot 2:** MSE loss vs iteration curve showing convergence of gradient descent
- Printed output:
  - Estimated intercept and slope for both methods
  - Difference from true values
  - Final explanation comment


## Requirements

```bash
pip install numpy matplotlib
```

## Usage

```bash
python assignment.py
```

## Learning Objectives

- Understand analytical vs iterative optimization
- Observe gradient descent convergence
- Compare computational approaches for linear regression