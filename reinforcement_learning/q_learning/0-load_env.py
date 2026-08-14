#!/usr/bin/env python3
"""Loads a FrozenLake environment."""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load a FrozenLake environment.

    Args:
        desc: Custom description of the map.
        map_name: Name of a predefined map.
        is_slippery: Whether the ice is slippery.

    Returns:
        The FrozenLake environment.
    """
    if desc is None and map_name is None:
        desc = gym.envs.toy_text.frozen_lake.generate_random_map(size=8)

    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )

    return env
