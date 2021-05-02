import gym
import cv2
import numpy as np
from baselines.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, TimeLimit
from collections import deque


class StackAtariWrapper(gym.Wrapper):
    def __init__(self, env, stack=4):
        super().__init__(env)
        self.stack = stack
        self.stack_windows = deque([], maxlen=self.stack)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = np.asarray(observation, dtype=np.float32) / 255.0

        self.stack_windows.clear()

        for _ in range(self.stack):
            self.stack_windows.append(observation)

        return self.obs()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = np.asarray(observation, dtype=np.float32) / 255.0
        self.stack_windows.append(observation)
        return self.obs(), reward, done, info

    def obs(self):
        return np.array(self.stack_windows)


def make_atari(env_id, skip=4, stack=4, max_episode_steps=None):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=skip)
    env = StackAtariWrapper(env, stack=stack)
    if max_episode_steps is not None:
        max_episode_steps //= skip
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
