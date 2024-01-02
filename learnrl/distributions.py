import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium.spaces import (
    Discrete,
    Box,
    MultiBinary,
    MultiDiscrete,
    Dict
)

from torch.distributions import Distribution
from torch.distributions import (
    Categorical,
    Bernoulli,
    MultivariateNormal
)


def dist_index(action_space):
    if isinstance(action_space, Discrete):
        return CategoricalDist()
    elif isinstance(action_space, Box):
        return None  # continuous via multivariate normal
    elif isinstance(action_space, MultiBinary):
        return None  # bernoulli
    elif isinstance(action_space, MultiDiscrete):
        return None  # multicategorical
    elif isinstance(action_space, Dict):
        return None  # not sure yet
    else:
        raise NotImplementedError 


class CategoricalDist(nn.Module):

    def __init__(self):
        super(CategoricalDist, self).__init__()

    # Relocate to base class?
    def forward(self, logits, actions):
        dist = Categorical(logits=logits)
        if actions is None:
            actions = dist.sample()
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, logprobs, entropy
    

class DiagGaussianDist(nn.Module):

    def forward(self, logits, actions=None):
        pass


class BernoulliDist(nn.Module):

    def forward(self, logits, actions=None):
        pass


class MultiCategoricalDist(nn.Module):

    def forward(self, logits, actions=None):
        pass