#!/usr/bin/env python3
"""Gaussian process prediction."""
import numpy as np


class GaussianProcess:
    """Represent a noiseless one-dimensional Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initialize the process from observed inputs and outputs."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Compute the radial basis function covariance kernel."""
        squared_distance = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1)
            - 2 * np.matmul(X1, X2.T)
        )
        squared_distance = np.maximum(squared_distance, 0)
        return self.sigma_f ** 2 * np.exp(
            -squared_distance / (2 * self.l ** 2)
        )

    def predict(self, X_s):
        """Predict using the GP model"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        solved_y = np.linalg.solve(self.K, self.Y)
        solved_kernel = np.linalg.solve(self.K, K_s)
        mean = np.matmul(K_s.T, solved_y).reshape(-1)
        covariance = K_ss - np.matmul(K_s.T, solved_kernel)
        variance = np.maximum(np.diag(covariance), 0)
        return mean, variance
