#!/usr/bin/env python3
"""Initialize centroids for K-means."""

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

    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    return np.random.uniform(minimum, maximum, size=(k, d))
