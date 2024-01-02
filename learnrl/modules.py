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


class AgentModule(nn.Module):

    def __init__(self, network, action_space):
        super(AgentModule, self).__init__()
        self.network = network
        self.action_dist = dist_index(action_space)

    def forward(self, states, actions=None):
        logits = self.network(states)
        actions, logprobs, entropy = self.action_dist(logits, actions=actions)
        return actions, logprobs, entropy
    

class CriticModule(nn.Module):

    def __init__(self, network):
        super(CriticModule, self).__init__()
        self.network = network

    def forward(self, states):
        return self.network(states)


class DiscriminatorModule(nn.Module):

    def __init__(self):
        super(DiscriminatorModule, self).__init__()

    def forward(self, x):
        pass