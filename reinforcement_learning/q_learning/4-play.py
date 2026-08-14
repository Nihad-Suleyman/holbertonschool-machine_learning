#!/usr/bin/env python3
"""Module for making a trained Q-learning agent play."""

import numpy as np


def play(env, Q, max_steps=100):
    """Play one episode using a trained Q-table.

    Args:
        env: FrozenLake environment instance.
        Q: Trained Q-table.
        max_steps: Maximum number of steps in the episode.

    Returns:
        total_rewards: Total reward received.
        rendered_outputs: List of rendered board states.
    """
    state, _ = env.reset()

    total_rewards = 0
    rendered_outputs = []

    rendered_outputs.append(env.render())

    for _ in range(max_steps):
        action = np.argmax(Q[state])

        state, reward, terminated, truncated, _ = env.step(action)

        total_rewards += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs
