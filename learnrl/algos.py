import torch as th
import torch.nn as nn


class PPO:

    def __init__(
        self,
        agent,
        critic,
        collector
    ):
        self.agent = agent
        self.critic = critic
        self.collector = collector

    def learn(self, total_timesteps):
        pass

    def loss(self):
        pass

    def calc_gae(self):
        pass