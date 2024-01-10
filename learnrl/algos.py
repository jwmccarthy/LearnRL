import torch as th
import torch.nn as nn


class PPO:

    def __init__(
        self,
        agent,
        critic,
        collector,
        epochs=8,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95
    ):
        self.agent = agent
        self.critic = critic
        self.collector = collector

        self.epochs = epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def learn(self, total_timesteps):
        t = 0

        while t < total_timesteps:
            buffer = self.collector.collect()

            # let critic evaluate states from rollout
            buffer.values = self.critic(buffer.states)
            buffer.next_values = self.critic(buffer.next_states)

            # TODO: Edit way buffers are handled
            # Option 1: Initialize these elsewhere and add to buffer via dict
            # Option 2: Pass buffers to modules and let them add "out_keys"
            buffer.advantages = th.zeros_like(buffer.rewards)

            self._calc_gae(buffer)

            for e in range(self.epochs):
                self._update(buffer)
                break

            break

    def _update(self, buffer):
        for b in self.collector.sample(self.batch_size):
            print(b)
            break


    def _calc_gae(self, buffer):
        adv = th.zeros((self.collector.num_envs), dtype=th.float32)

        for i in reversed(range(self.collector.size)):
            b = buffer[i]
            adv = b.rewards \
                + self.gamma * b.next_values * (1 - b.starts) \
                - b.values \
                + self.gamma * self.gae_lambda * adv * (1 - b.starts)
            buffer.advantages[i] = adv

        buffer.returns = buffer.advantages + buffer.values

    def _normalize_adv(self, adv):
        return (adv - adv.mean()) / (adv.std() + 1e-8)