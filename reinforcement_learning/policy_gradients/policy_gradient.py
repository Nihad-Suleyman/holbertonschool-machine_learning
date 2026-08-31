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
    return exp / np.sum(exp, axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """Compute the Monte Carlo policy gradient.

    Args:
        state: The current observation of the environment.
        weight: The weight matrix.

    Returns:
        A tuple of the selected action and its gradient.
    """
    probabilities = policy(state, weight)
    action = np.random.choice(weight.shape[1], p=probabilities)
    action_probabilities = -probabilities
    action_probabilities[action] += 1

    return action, np.outer(state, action_probabilities)
