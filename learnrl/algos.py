import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from tqdm import tqdm
from utils import batch_sample, stack_states


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
        val_coef=0.5,
        ent_coef=0.0,
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
        self.val_coef = val_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.params = list(self.agent.parameters()) + list(self.critic.parameters())
        self.optimizer = optimizer(self.params, lr=lr, eps=1e-5)

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

            self._pre_update(buffer)

            self._calc_gae(buffer)

            for e in range(self.epochs):
                for b in batch_sample(self.collector.buffer, self.batch_size):
                    self._update(b.flatten())

            time_elapsed = self.collector.num_envs * self.collector.size
            t += time_elapsed
            pbar.update(time_elapsed)

    def _pre_update(self, buffer):
        # let critic evaluate states from rollout
        with th.no_grad():
            buffer.values = self.critic(buffer.states)
            buffer.next_values = self.critic(buffer.next_states)
        
        # TODO: Include typing in buffer class to avoid conversions like this
        # bootstrap for term and not trunc
        boot_idx = ~buffer.terms.bool() & buffer.truncs.bool()
        buffer.rewards[boot_idx] = self.gamma * buffer.next_values[boot_idx]

    def _update(self, b):
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

        # entropy loss
        entropy_loss = -th.mean(entropy)

        # step optimizer
        total_loss = policy_loss \
                + self.val_coef * value_loss \
                + self.ent_coef * entropy_loss
        self.optimizer.zero_grad()
        total_loss.backward()

        # clip gradient norm
        th.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)

        self.optimizer.step()

    def _calc_gae(self, buffer):

        # TODO: Edit way buffers are handled
        # Option 1: Initialize these elsewhere and add to buffer via dict
        # Option 2: Pass buffers to modules and let them add "out_keys"
        buffer.advantages = th.zeros_like(buffer.rewards)

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


class GAIFO(PPO):

    def __init__(
        self,
        agent,
        critic,
        discriminator,
        collector,
        expert_demos,
        lr=3e-4,
        epochs=10,
        batch_size=64,
        eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        val_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        optimizer=opt.Adam
    ):
        super(GAIFO, self).__init__(
            agent=agent, 
            critic=critic, 
            collector=collector,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            eps=eps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            val_coef=val_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer
        )

        self.expert_demos = expert_demos

        self.discriminator = discriminator

        self.disc_optimizer = optimizer(self.discriminator.parameters(), lr=lr, eps=1e-5)

    def _pre_update(self, buffer):
        super()._pre_update(buffer)

        buffer.state_pairs = stack_states(buffer.states, buffer.next_states)

        self._update_discriminator(buffer)

        with th.no_grad():
            buffer.rewards = -th.log(self.discriminator(buffer.state_pairs))

    def _update_discriminator(self, buffer):  
        loss_func = nn.BCELoss()

        for e in range(self.epochs):
            for exp_data, agt_data in zip(batch_sample(self.expert_demos, self.batch_size),
                                          batch_sample(buffer.state_pairs, self.batch_size)):
                self.disc_optimizer.zero_grad()

                # train on expert data

                exp_pred = self.discriminator(exp_data)
                exp_loss = loss_func(exp_pred, th.zeros_like(exp_pred))

                # train on agent data
                agt_pred = self.discriminator(agt_data)
                agt_loss = loss_func(agt_pred, th.ones_like(agt_pred))

                # backpropagate
                disc_loss = exp_loss + agt_loss
                disc_loss.backward()
                self.disc_optimizer.step()