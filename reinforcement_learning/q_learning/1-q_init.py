#!/usr/bin/env python3
"""Module for initializing a Q-table."""

import numpy as np


def q_init(env):
    """Initialize the Q-table.

    Args:
        env: FrozenLake environment instance.

    Returns:
        Q-table initialized with zeros.
    """
    return np.zeros(
        (env.observation_space.n, env.action_space.n)
    )
