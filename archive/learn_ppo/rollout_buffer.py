import numpy as np
import torch as th

from typing import NamedTuple


class RolloutSample(NamedTuple):
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
        gae_gamma,
        gae_lambda
    ):  
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda

        self.step = 0

        # initialize buffers
        self.observations = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.float32)
        self.values = np.zeros((self.buffer_size), dtype=np.float32)
        self.logprobs = np.zeros((self.buffer_size), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size), dtype=np.float32)

    def add(
        self,
        observation,
        action,
        reward,
        done,
        value,
        logprob
    ):
        self.observations[self.step] = np.array(observation)
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value.clone().cpu().numpy()
        self.logprobs[self.step] = logprob.clone().cpu().numpy()

        self.step += 1

    def sample(self, batch_size):
        self.step = 0  # sampling induces buffer reset

        random_inds = np.random.permutation(self.buffer_size)

        for batch_start in range(0, self.buffer_size, batch_size):
            batch_end = batch_start + batch_size
            batch_inds = random_inds[batch_start:batch_end]
            yield self._create_sample(batch_inds)

    def _create_sample(self, batch_inds):
        sample = list(map(th.as_tensor, (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.returns[batch_inds],
            self.values[batch_inds],
            self.logprobs[batch_inds],
            self.advantages[batch_inds]
        )))
        return RolloutSample(*sample)

    def compute_advantages(self, last_value, last_done):
        next_done = last_done
        next_value = last_value.clone().cpu().numpy()

        advantage = 0

        for t in reversed(range(self.buffer_size)):
            delta = self.rewards[t] + self.gae_gamma * next_value * (1 - next_done) - self.values[t]
            advantage = delta + self.gae_gamma * self.gae_lambda * advantage * (1 - next_done)
            next_done = self.dones[t]
            next_value = self.values[t]
            self.advantages[t] = advantage

        self.returns = self.advantages + self.values