#!/usr/bin/env python3
"""We will look at likelihood of some disease"""
import numpy as np


def likelihood(x, n, P):
    """we will use bayesian"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError('x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.any((P < 0) | (P > 1)):
        raise ValueError('All values in P must be in range [0, 1]')
    factn, factk, factnk = 1, 1, 1
    for i in range(1, n + 1):
            factn *= i
    for i in range(1, x + 1):
            factk *= i
    for i in range(1, n + 1 - x):
            factnk *= i
    return list(factn / (factk * factnk) * p ** x * \
            (1 - p) ** (n - x) for p in P)