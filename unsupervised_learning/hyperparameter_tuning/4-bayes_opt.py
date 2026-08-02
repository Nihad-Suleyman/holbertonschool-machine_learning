#!/usr/bin/env python3
"""Bayesian optimization with expected improvement."""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Optimize a black-box function with a Gaussian process."""

    def __init__(self,
                 f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize the optimizer and acquisition sample grid."""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], num=ac_samples
        ).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Acquisition function"""
        mean, variance = self.gp.predict(self.X_s)
        standard_deviation = np.sqrt(np.maximum(variance, 0))

        if self.minimize:
            improvement = np.min(self.gp.Y) - mean - self.xsi
        else:
            improvement = mean - np.max(self.gp.Y) - self.xsi

        z_score = np.zeros_like(standard_deviation)
        has_uncertainty = standard_deviation > 0
        z_score[has_uncertainty] = (
            improvement[has_uncertainty]
            / standard_deviation[has_uncertainty]
        )
        expected_improvement = (
            improvement * norm.cdf(z_score)
            + standard_deviation * norm.pdf(z_score)
        )
        expected_improvement[~has_uncertainty] = 0

        X_next = self.X_s[np.argmax(expected_improvement)]

        return X_next, expected_improvement
