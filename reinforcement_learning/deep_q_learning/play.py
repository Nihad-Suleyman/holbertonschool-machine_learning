#!/usr/bin/env python3
"""Play Atari Breakout using a trained DQN agent."""

import gymnasium as gym
import numpy as np

from PIL import Image

from keras.models import load_model
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from rl.core import Processor


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class GymnasiumWrapper(gym.Wrapper):
    """Make Gymnasium compatible with keras-rl2."""

    def reset(self, **kwargs):
        """Reset environment using old Gym API."""
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple):
            observation, _ = result
            return observation

        return result

    def step(self, action):
        """Convert Gymnasium step output to old Gym API."""
        result = self.env.step(action)

        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
            return observation, reward, done, info

        return result

    def render(self, mode="human"):
        """Render the game."""
        return self.env.render()


class AtariProcessor(Processor):
    """Preprocess Atari frames."""

    def process_observation(self, observation):
        """Resize observation to 84x84 grayscale."""
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE)
        img = img.convert("L")

        processed_observation = np.array(img)

        return processed_observation.astype("uint8")

    def process_state_batch(self, batch):
        """Normalize states."""
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        """Clip rewards."""
        return np.clip(reward, -1.0, 1.0)


def make_environment():
    """Create rendered Breakout environment."""
    try:
        env = gym.make(
            "ALE/Breakout-v5",
            render_mode="human",
            frameskip=1,
            repeat_action_probability=0.0
        )
    except Exception:
        try:
            env = gym.make(
                "BreakoutNoFrameskip-v4",
                render_mode="human"
            )
        except TypeError:
            env = gym.make("BreakoutNoFrameskip-v4")

    return GymnasiumWrapper(env)


def main():
    """Load policy and play Breakout."""
    env = make_environment()

    nb_actions = env.action_space.n

    model = load_model("policy.h5")

    memory = SequentialMemory(
        limit=1000000,
        window_length=WINDOW_LENGTH
    )

    policy = GreedyQPolicy()

    processor = AtariProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=memory,
        processor=processor,
        nb_steps_warmup=0,
        gamma=0.99,
        target_model_update=10000
    )

    dqn.compile(
        Adam(learning_rate=0.00025),
        metrics=["mae"]
    )

    dqn.test(
        env,
        nb_episodes=5,
        visualize=True
    )

    env.close()


if __name__ == "__main__":
    main()
