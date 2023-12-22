import attr
import numpy as np
import torch as th


# attrs tensor converter
tensor_ib = attr.ib(converter=th.as_tensor)


class RolloutBuffer():

    @attr.s(slots=True, converter=tensor_ib)
    class RolloutSample:
        states:     th.Tensor = tensor_ib
        actions:    th.Tensor = tensor_ib
        starts:     th.Tensor = tensor_ib
        returns:    th.Tensor = tensor_ib
        values:     th.Tensor = tensor_ib
        logprobs:   th.Tensor = tensor_ib
        advantages: th.Tensor = tensor_ib
    
    def __init__(
        self,
        size,
        state_dim, 
        action_dim,
        num_envs,
        gamma=0.99,
        gae_lambda=0.95
    ):
        self.step = 0
        self.size = size
        self.num_envs = num_envs

        # adv hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # initialize numpy array buffers
        self.states     = np.array((self.size, self.num_envs, state_dim),
                                    dtype=np.float32)
        self.actions    = np.array((self.size, self.num_envs, action_dim), 
                                   dtype=np.float32)
        self.rewards    = np.array((self.size, self.num_envs), dtype=np.float32)
        self.starts     = np.array((self.size, self.num_envs), dtype=np.float32)
        self.returns    = np.array((self.size, self.num_envs), dtype=np.float32)
        self.values     = np.array((self.size, self.num_envs), dtype=np.float32)
        self.logprobs   = np.array((self.size, self.num_envs), dtype=np.float32)
        self.advantages = np.array((self.size, self.num_envs), dtype=np.float32)

    def append(
        self,
        states,
        actions,
        rewards,
        values,
        logprobs
    ):
        self.states[self.step] = states
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.values[self.step] = values.cpu().numpy()
        self.logprobs[self.step] = logprobs.cpu().numpy()

        self.step += 1

    def sample(self, batch_size):
        self.step = 0  # sampling induces buffer reset

        random_idx = np.random.permutation(self.size)

        for batch_start in range(0, self.size, batch_size):
            batch_end = batch_start + batch_size
            batch_idx = random_idx[batch_start:batch_end]
            yield self.RolloutSample(
                self.states[batch_idx],
                self.actions[batch_idx],
                self.returns[batch_idx],
                self.values[batch_idx],
                self.logprobs[batch_idx],
                self.advantages[batch_idx]
            )
            
    def compute_advantages(self, last_starts, last_values):
        next_starts = last_starts
        next_values = last_values.flatten().cpu().numpy()
        advantages  = np.zeros((self.num_envs), dtype=np.float32)

        for t in reversed(range(self.size)):
            delta = self.rewards[t] + self.gamma * next_values * (1 - next_starts) - self.values[t]
            advantages = delta + self.gamma * self.gae_lambda * advantages * (1 - next_starts)
            self.advantages[t] = advantages
            next_starts = self.starts[t]
            next_values = self.values[t]

        self.returns = self.advantages + self.values


class StatePairRolloutBuffer():

    @attr.s(slots=True)
    class StatePairSample:
        states:     th.Tensor = tensor_ib
        starts:     th.Tensor = tensor_ib
        returns:    th.Tensor = tensor_ib
        values:     th.Tensor = tensor_ib
        logprobs:   th.Tensor = tensor_ib
        advantages: th.Tensor = tensor_ib