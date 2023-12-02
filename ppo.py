import numpy as np

from tqdm import tqdm

import torch as th
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical

from rollout_buffer import RolloutBuffer


class Policy(nn.Module):

    def __init__(self, obs_dim, act_dim, learning_rate=3e-4, eps=1e-5):

        super(Policy, self).__init__()
        
        # value net
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # actor net
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )

        self.optimizer = Adam(self.parameters(), lr=learning_rate, eps=eps)

    def get_value(self, obs):
        obs_tensor = th.as_tensor(obs)
        with th.no_grad():
            value = self.critic(obs_tensor)
        return value.item()

    def get_action_dist(self, obs):
        logits = self.actor(obs)
        return Categorical(logits=logits)

    def evaluate_actions(self, obs, action=None):
        dist = self.get_action_dist(obs)
        value = self.critic(obs)

        # sample action if not provided
        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, value, logprob, entropy


class PPO():

    def __init__(
        self,
        env,
        learning_rate=3e-4,
        rollout_size=2048,
        gae_gamma=0.99,
        gae_lambda=0.95,
        minibatch_size=64,
        training_epochs=10,
        epsilon=0.2,
        val_coef=0.5,
        max_grad_norm=0.5
    ):
        self.env = env

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n

        self.policy = Policy(self.obs_dim, self.act_dim, learning_rate=learning_rate)

        self.rollout_size = rollout_size
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        
        self.rollout_buffer = RolloutBuffer(
            self.rollout_size, self.obs_dim, self.act_dim, self.gae_gamma, self.gae_lambda
        )

        self.minibatch_size = minibatch_size
        self.training_epochs = training_epochs
        self.epsilon = epsilon
        self.val_coef = val_coef
        self.max_grad_norm = max_grad_norm

        self.reward_means = []
        self.length_means = []

    def collect_rollout(self):
        last_done = 0
        last_obs = self.env.reset()[0]

        ep_length, ep_reward = 0, 0
        ep_rewards, ep_lengths = [], []
        
        for t in range(self.rollout_size):

            with th.no_grad():
                obs_tensor = th.as_tensor(last_obs)
                action, value, logprob, _ = self.policy.evaluate_actions(obs_tensor)
            action = action.cpu().numpy()

            next_obs, reward, term, trunc, _ = self.env.step(action)
            next_done = term or trunc

            if next_done and not trunc:
                term_val = self.policy.get_value(last_obs)
                reward += self.gae_gamma * term_val

            self.rollout_buffer.add(
                last_obs,
                action,
                reward,
                last_done,
                value,
                logprob
            )
            
            ep_length += 1
            ep_reward += reward

            # reset env if episode done
            if next_done:
                ep_lengths.append(ep_length)
                ep_rewards.append(ep_reward)
                ep_length, ep_reward = 0, 0
                next_obs = self.env.reset()[0]

            last_done = next_done
            last_obs = next_obs

        # bootstrap in event of timeout
        with th.no_grad():
            obs_tensor = th.as_tensor(last_obs)
            last_value = self.policy.critic(obs_tensor)
        self.rollout_buffer.compute_advantages(last_value, last_done)

        return np.mean(ep_lengths), np.mean(ep_rewards)

    def update(self):
        for _ in range(self.training_epochs):
            for minibatch in self.rollout_buffer.sample(self.minibatch_size):
                observations = minibatch.observations
                actions = minibatch.actions.long().flatten()

                _, values, logprobs, _ = self.policy.evaluate_actions(observations, action=actions)
                values = values.flatten()

                # normalize advantages
                advantages = minibatch.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # logprob ratios
                ratios = th.exp(logprobs - minibatch.logprobs)

                # surrogate loss
                policy_loss1 = advantages * ratios
                policy_loss2 = advantages * th.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon)
                policy_loss = -th.min(policy_loss1, policy_loss2).mean()

                # value loss
                value_loss = F.mse_loss(minibatch.returns, values)

                # total loss
                total_loss = policy_loss + self.val_coef * value_loss

                self.policy.optimizer.zero_grad()
                total_loss.backward()
                # Clip grad norm
                norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # print(norm)
                # for param in self.policy.parameters():
                #     print(param.grad.shape)                
                self.policy.optimizer.step()

    def learn(self, total_timesteps):
        num_updates = total_timesteps // self.rollout_size

        pbar = tqdm(range(num_updates))
        for _ in pbar:
            mean_ep_lengths, mean_ep_rewards = self.collect_rollout()

            self.reward_means.append(mean_ep_rewards)
            self.length_means.append(mean_ep_lengths)

            mean_rew = round(np.mean(self.reward_means), 2)
            mean_len = round(np.mean(self.length_means), 2)
            
            pbar.set_description(f"{mean_len} : {mean_rew}")
            self.update()

    def predict(self, obs):
        obs_tensor = th.as_tensor(obs)
        with th.no_grad():
            dist = self.policy.get_action_dist(obs_tensor)
            action = th.argmax(dist.probs)
        return action.cpu().numpy()