#!/usr/bin/env python3
"""Policy gradient training module."""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def _reset(env):
    """Reset an environment and return only the observation."""
    state = env.reset()
    if isinstance(state, tuple):
        return state[0]
    return state


def _step(env, action):
    """Take one environment step across Gym/Gymnasium APIs."""
    result = env.step(action)
    if len(result) == 5:
        state, reward, terminated, truncated, _ = result
        return state, reward, terminated or truncated

    state, reward, done, _ = result
    return state, reward, done


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Implement full policy gradient training.

    Args:
        env: The initial environment.
        nb_episodes: The number of episodes used for training.
        alpha: The learning rate.
        gamma: The discount factor.

    Returns:
        A list containing the score of all rounds during training.
    """
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state = _reset(env)
        episode_gradients = []
        episode_rewards = []
        done = False

        while not done:
            action, gradient = policy_gradient(state, weight)
            state, reward, done = _step(env, action)

            episode_gradients.append(gradient)
            episode_rewards.append(reward)

        score = sum(episode_rewards)
        scores.append(score)

        cumulative_reward = 0
        for step in reversed(range(len(episode_rewards))):
            cumulative_reward *= gamma
            cumulative_reward += episode_rewards[step]
            weight += alpha * episode_gradients[step] * cumulative_reward

        print("Episode: {} Score: {}".format(episode, score))

    return scores
