import numpy as np
import gym


class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward=-0.1, max_reward=0.1):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)
