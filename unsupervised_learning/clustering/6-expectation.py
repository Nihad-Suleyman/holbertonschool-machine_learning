#!/usr/bin/env python3
"""Expectation step for GMM."""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None

    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None

    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape != (k, d) or S.shape != (k, d, d):
        return None, None

    if not np.isclose(np.sum(pi), 1):
        return None, None

    try:
        g = np.zeros((k, n))

        for i in range(k):
            g[i] = pi[i] * pdf(X, m[i], S[i])

        total = np.sum(g, axis=0)

        if np.any(total == 0):
            return None, None

        g /= total
        op = np.sum(np.log(total))

        return g, op

    except Exception:
        return None, None
