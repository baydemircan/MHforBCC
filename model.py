
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_likelihood(beta, X, y):
    z = X @ beta
    p = sigmoid(z)
    epsilon = 1e-9  # To prevent log(0)
    likelihood = np.sum(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))
    return likelihood

def prior(beta, lambda_reg=1.0):
    sigma_squared = 1  # Variance of the prior
    l2_penalty = lambda_reg * np.sum(beta[1:]**2)  # L2 regularization, excluding intercept
    return -0.5 * np.sum(beta ** 2) / sigma_squared - l2_penalty