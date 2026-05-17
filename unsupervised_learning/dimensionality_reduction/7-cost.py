#!/usr/bin/env python3
"""Cost module."""

import numpy as np


def cost(P, Q):
    """
    Calculates the cost of the t-SNE transformation.
    """
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    C = np.sum(P * np.log(P / Q))

    return C
