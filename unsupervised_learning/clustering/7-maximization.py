#!/usr/bin/env python3
"""Maximization step for GMM."""

import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k, n2 = g.shape

    if n != n2:
        return None, None, None

    try:
        Nk = np.sum(g, axis=1)

        pi = Nk / n

        m = (g @ X) / Nk[:, np.newaxis]

        S = np.zeros((k, d, d))

        for i in range(k):
            diff = X - m[i]
            S[i] = (g[i, :, np.newaxis] * diff).T @ diff / Nk[i]

        return pi, m, S

    except Exception:
        return None, None, None
