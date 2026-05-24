#!/usr/bin/env python3
"""Calculate intra-cluster variance."""

import numpy as np


def variance(X, C):
    """Calculates total intra-cluster variance."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    n, d = X.shape
    k, d2 = C.shape

    if d != d2:
        return None

    try:
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        closest = np.min(distances, axis=1) ** 2
        return np.sum(closest)

    except Exception:
        return None
