#!/usr/bin/env python3
"""Updatable Gaussian process."""
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

    def update(self, X_new, Y_new):
        """Update the GP model"""
        X_new = np.asarray(X_new).reshape(1, self.X.shape[1])
        Y_new = np.asarray(Y_new).reshape(1, self.Y.shape[1])
        self.X = np.concatenate((self.X, X_new), axis=0)
        self.Y = np.concatenate((self.Y, Y_new), axis=0)
        self.K = self.kernel(self.X, self.X)
