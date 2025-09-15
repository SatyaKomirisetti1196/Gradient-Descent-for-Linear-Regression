import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate synthetic data
n_samples = 200
X = np.linspace(0, 5, n_samples)
true_intercept = 3
true_slope = 4
noise = np.random.normal(0, 1, n_samples)  # Gaussian noise
y = true_intercept + true_slope * X + noise

# Plot the raw data
plt.figure(figsize=(12, 5))

# First subplot: data and regression lines
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.7, label='Raw data', color='blue')
plt.title('Synthetic Data: y = 3 + 4x + ε')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# Prepare design matrix with bias term (column of ones)
X_b = np.c_[np.ones((n_samples, 1)), X]  # Add x0 = 1 to each instance

# 2. Closed-form solution (Normal Equation)
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
closed_form_intercept = theta_best[0]
closed_form_slope = theta_best[1]

print("Closed-form solution:")
print(f"Intercept: {closed_form_intercept:.4f}, Slope: {closed_form_slope:.4f}")

# Plot the closed-form fitted line
y_pred_closed = X_b.dot(theta_best)
plt.plot(X, y_pred_closed, 'r-', linewidth=2, label='Closed-form solution')

# 3. Gradient Descent implementation
# Hyperparameters
eta = 0.05  # Learning rate
n_iterations = 1000
m = len(X_b)  # Number of samples

# Initialize parameters
theta = np.zeros(2)  # [intercept, slope]
loss_history = np.zeros(n_iterations)

# Gradient Descent
for iteration in range(n_iterations):
    # Compute predictions and errors
    predictions = X_b.dot(theta)
    errors = predictions - y
    
    # Compute gradient (using the formula: (2/m) * X_b.T.dot(errors))
    gradients = (2/m) * X_b.T.dot(errors)
    
    # Update parameters
    theta = theta - eta * gradients
    
    # Compute and store MSE loss
    loss = np.mean(errors**2)
    loss_history[iteration] = loss

# Print final parameters from Gradient Descent
gd_intercept = theta[0]
gd_slope = theta[1]
print("\nGradient Descent solution:")
print(f"Intercept: {gd_intercept:.4f}, Slope: {gd_slope:.4f}")

# Plot the Gradient Descent fitted line
y_pred_gd = X_b.dot(theta)
plt.plot(X, y_pred_gd, 'g--', linewidth=2, label='Gradient Descent')
plt.legend()

# Second subplot: loss curve
plt.subplot(1, 2, 2)
plt.plot(range(n_iterations), loss_history)
plt.title('Loss Curve (MSE vs Iterations)')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.grid(True)

plt.tight_layout()
plt.show()

# 4. Comparison
print("\nComparison:")
print(f"True parameters: Intercept = {true_intercept}, Slope = {true_slope}")
print(f"Closed-form: Intercept = {closed_form_intercept:.4f}, Slope = {closed_form_slope:.4f}")
print(f"Gradient Descent: Intercept = {gd_intercept:.4f}, Slope = {gd_slope:.4f}")
print(f"Difference: Intercept = {abs(closed_form_intercept - gd_intercept):.6f}, Slope = {abs(closed_form_slope - gd_slope):.6f}")

# Short explanation
print("\nExplanation:")
print("Both the closed-form solution and gradient descent converged to the same parameters,")
print("which demonstrates the effectiveness of gradient descent for linear regression.")
print("The small difference between the true parameters and estimated parameters is due")
print("to the Gaussian noise added to the synthetic data.")