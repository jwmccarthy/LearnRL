import torch as th
import torch.nn as nn


class PPO:

    def __init__(
        self,
        agent,
        critic,
        collector,
        gamma=0.99,
        gae_lambda=0.95
    ):
        self.agent = agent
        self.critic = critic
        self.collector = collector

        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def learn(self, total_timesteps):
        # TODO: Probably no need for setting buffer attribute
        # Passing by reference to local-scope buffer var is fine
        self.buffer = self.collector.collect()

        # let critic evaluate states from rollout
        self.buffer.values = self.critic(self.buffer.states)
        self.buffer.next_values = self.critic(self.buffer.next_states)

        # TODO: Edit way buffers are handled
        # Option 1: Initialize these elsewhere and add to buffer via dict
        # Option 2: Pass buffers to modules and let them add "out_keys"
        self.buffer.advantages = th.zeros_like(self.buffer.rewards)

        self._calc_gae()

    def _update(self):
        pass

    def _calc_gae(self):
        adv = th.zeros((self.collector.num_envs), dtype=th.float32)

        # Incorporate boostrapping on termination here?
        for i in reversed(range(self.collector.size)):
            b = self.buffer[i]
            adv = b.rewards \
                + self.gamma * b.next_values * (1 - b.starts) \
                - b.values \
                + self.gamma * self.gae_lambda * adv * (1 - b.starts)
            
            self.buffer.advantages[i] = adv

        self.buffer.returns = self.buffer.advantages + self.buffer.values
