import torch as th
import torch.nn as nn
import torch.optim as opt
import gymnasium as gym

from distributions import dist_index


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
        return self.network(states).squeeze(dim=-1)


class DiscriminatorModule(nn.Module):

    def __init__(self, network):
        super(DiscriminatorModule, self).__init__()
        self.network = network

    def forward(self, pairs):
        return self.network(pairs).squeeze(dim=-1)