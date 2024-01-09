import torch as th
import torch.nn as nn
import gymnasium as gym

from distributions import dist_index


class AgentModule(nn.Module):

    def __init__(self, network, action_space, in_key="states"):
        super(AgentModule, self).__init__()
        self.network = network
        self.in_key = in_key
        self.action_dist = dist_index(action_space)

    def forward(self, buffer, actions=None):
        logits = self.network(buffer[self.in_key])
        actions, logprobs, entropy = self.action_dist(logits, actions=actions)
        return actions, logprobs, entropy
    

class CriticModule(nn.Module):

    def __init__(self, network, in_key="states", out_key="values"):
        super(CriticModule, self).__init__()
        self.network = network
        self.in_key = in_key
        self.out_key = out_key

    def forward(self, buffer):
        buffer[self.out_key] = self.network(buffer[self.in_key])


class DiscriminatorModule(nn.Module):

    def __init__(self):
        super(DiscriminatorModule, self).__init__()

    def forward(self, x):
        pass