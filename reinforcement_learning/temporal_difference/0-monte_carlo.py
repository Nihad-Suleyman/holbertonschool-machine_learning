#!/usr/bin/env python3
"""Monte Carlo algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm.

    Args:
        env: environment instance
        V: numpy.ndarray containing value estimates
        policy: function that takes a state and returns an action
        episodes: total number of episodes
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """

    for _ in range(episodes):
        episode = []

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        for _ in range(max_steps):
            action = policy(state)

            result = env.step(action)

            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            episode.append((state, reward))
            state = next_state

            if done:
                break

        for i, (state, _) in enumerate(episode):
            previous_states = [step[0] for step in episode[:i]]

            if state not in previous_states:
                G = 0

                for j in range(i, len(episode)):
                    reward = episode[j][1]
                    G += (gamma ** (j - i)) * reward

                V[state] += alpha * (G - V[state])

    return V
