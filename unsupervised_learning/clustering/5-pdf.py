#!/usr/bin/env python3
"""Calculate the PDF of a Gaussian distribution."""

import numpy as np


def pdf(X, m, S):
    """Calculates the probability density function."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None

    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d or S.shape != (d, d):
        return None

    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)

        if det <= 0:
            return None

        diff = X - m
        exponent = -0.5 * np.sum((diff @ inv) * diff, axis=1)

        coeff = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        P = coeff * np.exp(exponent)

        return np.maximum(P, 1e-300)

    except Exception:
        return None
