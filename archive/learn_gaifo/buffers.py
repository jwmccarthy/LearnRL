import numpy as np
import torch as th

from dataclasses import dataclass


@dataclass
class Sample:
    observations: th.Tensor
    actions: th.Tensor
    returns: th.Tensor
    values: th.Tensor
    logprobs: th.Tensor
    advantages: th.Tensor


class RolloutBuffer():

    def __init__(
        self,
        buffer_size,
        obs_dim, 
        act_dim,
        num_envs,
        gamma=0.99,
        gae_lambda=0.95
    ):
        # data dimensions
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_envs = num_envs

        # hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.step = 0

        # initialize buffers
        self.observations = np.zeros((self.buffer_size, self.num_envs, self.obs_dim), dtype=np.float32)
        self.final_obs = np.zeros((self.buffer_size, self.num_envs, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.starts = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.logprobs = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.num_envs), dtype=np.float32)

    def add(
        self,
        observation,
        final_obs,
        action,
        reward,
        start,
        value,
        logprob
    ):
        self.observations[self.step] = np.array(observation)
        self.final_obs[self.step] = np.array(final_obs)
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.starts[self.step] = start
        self.values[self.step] = value.cpu().numpy().squeeze()
        self.logprobs[self.step] = logprob.cpu().numpy().squeeze()

        self.step += 1

    def sample(self, batch_size):
        self.step = 0  # sampling induces buffer reset

        random_idx = np.random.permutation(self.buffer_size)

        for batch_start in range(0, self.buffer_size, batch_size):
            batch_end = batch_start + batch_size
            batch_idx = random_idx[batch_start:batch_end]
            yield self._create_sample(batch_idx)

    def _create_sample(self, batch_idx):
        sample = list(map(th.as_tensor, (
            self.observations[batch_idx],
            self.actions[batch_idx],
            self.returns[batch_idx],
            self.values[batch_idx],
            self.logprobs[batch_idx],
            self.advantages[batch_idx]
        )))
        return Sample(*sample)

    def compute_advantages(self, last_value, last_start):
        next_start = last_start
        next_value = last_value.cpu().numpy().flatten()

        advantage = np.zeros((self.num_envs))

        for t in reversed(range(self.buffer_size)):
            delta = self.rewards[t] + self.gamma * next_value * (1 - next_start) - self.values[t]
            advantage = delta + self.gamma * self.gae_lambda * advantage * (1 - next_start)
            self.advantages[t] = advantage
            next_start = self.starts[t]
            next_value = self.values[t]

        self.returns = self.advantages + self.values