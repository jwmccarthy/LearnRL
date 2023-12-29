import torch as th
import torch.nn as nn
import gymnasium as gym

from distributions import dist_index


def dims_to_sequential(in_dim, hidden_dims, out_dim, act_func):
    network = nn.Sequential()

    last_dim = in_dim
    for next_dim in hidden_dims:
        network.append(nn.Linear(last_dim, next_dim))
        network.append(act_func())
        last_dim = next_dim
    network.append(nn.Linear(last_dim, out_dim))
    
    return network


class StochasticAgent(nn.Module):

    def __init__(
        self,
        env,
        act_func=nn.ReLU,
        actor_dims=[64, 64],
        critic_dims=[64, 64]
    ):
        # initialize action distribution
        self.action_dist = dist_index(env.single_action_space)

        # input/output dimensions (TODO #1)
        self.state_dim = gym.spaces.flatdim(env.single_observation_space)
        self.action_dim = self.action_dist.action_dim

        # initialize actor network
        self.actor = dims_to_sequential(
            self.state_dim, actor_dims, self.action_dim, act_func
        )

        # initialize critic network
        self.critic = dims_to_sequential(
            self.state_dim, critic_dims, 1, act_func
        )

    def get_value(self, states):
        states_tensor = th.as_tensor(states)
        return self.critic(states_tensor)

    def forward(self, states, actions=None):
        logits = self.actor(states)
        values = self.critic(states)
        actions, logprobs, entropy = self.action_dist(logits, actions=actions)
        return actions, values, logprobs, entropy