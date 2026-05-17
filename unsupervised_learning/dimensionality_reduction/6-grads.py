#!/usr/bin/env python3
"""Gradients module."""

import numpy as np

Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    Calculates the gradients of Y.
    """
    n, ndim = Y.shape

    Q, num = Q_affinities(Y)

    dY = np.zeros((n, ndim))

    for i in range(n):
        diff = Y[i] - Y

        dY[i] = np.sum(
            ((P[i] - Q[i]) * num[i]).reshape((n, 1)) * diff,
            axis=0
        )

    return dY, Q
