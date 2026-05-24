#!/usr/bin/env python3
"""Find best number of clusters using BIC."""

import numpy as np

expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None

    n, d = X.shape

    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None

    if kmin >= kmax:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    results = []
    log_likelihoods = []
    bics = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, l = expectation_maximization(
            X, k, iterations, tol, verbose
        )

        if pi is None:
            return None, None, None, None

        results.append((pi, m, S))
        log_likelihoods.append(l)

        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)
        bic = p * np.log(n) - 2 * l
        bics.append(bic)

    l = np.array(log_likelihoods)
    b = np.array(bics)

    best_index = np.argmin(b)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, l, b
