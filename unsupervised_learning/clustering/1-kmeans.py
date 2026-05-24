#!/usr/bin/env python3
"""K-means clustering."""

import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    if k > n:
        return None

    return np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))


def kmeans(X, k, iterations=1000):
    """Perform K-means on a dataset."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    if k > n:
        return None, None

    C = initialize(X, k)

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        C_new = C.copy()

        for j in range(k):
            if np.any(clss == j):
                C_new[j] = np.mean(X[clss == j], axis=0)
            else:
                C_new[j] = np.random.uniform(np.min(X, axis=0),
                                             np.max(X, axis=0),
                                             size=(d,))

        if np.array_equal(C, C_new):
            return C_new, clss

        C = C_new

    distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
