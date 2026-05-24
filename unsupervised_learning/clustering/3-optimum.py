#!/usr/bin/env python3
"""Find optimum number of clusters."""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None

    if kmin >= kmax:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None:
            return None, None

        results.append((C, clss))
        variances.append(variance(X, C))

    base_variance = variances[0]
    d_vars = [base_variance - var for var in variances]

    return results, d_vars
