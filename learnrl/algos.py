import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from tqdm import tqdm


class PPO:

    def __init__(
        self,
        agent,
        critic,
        collector,
        lr=3e-4,
        epochs=10,
        batch_size=64,
        eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        optimizer=opt.Adam
    ):
        self.agent = agent
        self.critic = critic
        self.collector = collector

        self.epochs = epochs
        self.batch_size = batch_size

        self.eps = eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm

        self.params = list(self.agent.parameters()) + list(self.critic.parameters())
        self.optimizer = optimizer(self.params, lr=lr)

    def learn(self, total_timesteps):
        t = 0

        ep_rewards = []
        ep_lengths = []

        pbar = tqdm(total=total_timesteps)
        while t < total_timesteps:
            buffer, mean_rew, mean_len = self.collector.collect()
            ep_rewards.append(mean_rew)
            ep_lengths.append(mean_len)
            pbar.set_description(f"{np.mean(ep_rewards):<3.2f} : {np.mean(ep_lengths):<3.2f}")

            # let critic evaluate states from rollout
            with th.no_grad():
                buffer.values = self.critic(buffer.states)
                buffer.next_values = self.critic(buffer.next_states)

            # TODO: Edit way buffers are handled
            # Option 1: Initialize these elsewhere and add to buffer via dict
            # Option 2: Pass buffers to modules and let them add "out_keys"
            buffer.advantages = th.zeros_like(buffer.rewards)

            self._calc_gae(buffer)

            # print(buffer.rewards.sum(dim=0))

            for e in range(self.epochs):
                self._update()

            time_elapsed = self.collector.num_envs * self.collector.size
            t += time_elapsed
            pbar.update(time_elapsed)

    def _update(self):
        for b in self.collector.sample(self.batch_size):
            # evaluate states, actions
            _, logprobs, entropy = self.agent(b.states, actions=b.actions)
            values = self.critic(b.states)

            # policy loss
            ratios = th.exp(logprobs - b.logprobs)
            advantages = self._normalize_adv(b.advantages)
            policy_loss = -th.min(
                advantages * ratios,
                advantages * th.clamp(ratios, 1 - self.eps, 1 + self.eps)
            ).mean()

            # value loss
            value_loss = F.mse_loss(b.returns, values)

            # step optimizer
            total_loss = policy_loss + 0.5 * value_loss
            self.optimizer.zero_grad()
            total_loss.backward()

            # clip gradient norm
            th.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)

            self.optimizer.step()

    def _calc_gae(self, buffer):
        adv = th.zeros((self.collector.num_envs), dtype=th.float32)

        # Incorporate boostrapping on termination here?
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
