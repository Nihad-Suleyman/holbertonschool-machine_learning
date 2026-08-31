#!/usr/bin/env python3
"""Policy gradient module."""

import numpy as np


def policy(matrix, weight):
    """Compute a policy using a weight matrix.

    Args:
        matrix: The state matrix.
        weight: The weight matrix.

    Returns:
        A numpy.ndarray containing the policy probabilities.
    """
    z = np.matmul(matrix, weight)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)
