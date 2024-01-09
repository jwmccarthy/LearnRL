import torch as th
from gymnasium.vector import SyncVectorEnv
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    flatdim
)
from utils import to_tensor


def logit_dim(action_space):
    if isinstance(action_space, MultiDiscrete):
        return action_space.nvec
    else: 
        return action_space.n


class SyncTorchEnv(SyncVectorEnv):

    def __init__(
        self,
        env_fns,
        observation_space=None,
        action_space=None,
        copy=True
    ):
        super(SyncTorchEnv, self).__init__(
            env_fns,
            observation_space,
            action_space,
            copy
        )

        self.flat_dim = flatdim(self.single_observation_space)
        self.state_dim = self.single_observation_space.shape
        self.action_dim = self.single_action_space.shape
        self.logit_dim = logit_dim(self.single_action_space)

    def step(self, actions):
        states, rewards, terms, truncs, infos = super().step(
            actions.cpu().numpy()
        )
        
        return (
            to_tensor(states),
            to_tensor(rewards),
            to_tensor(terms, dtype=th.bool),
            to_tensor(truncs, dtype=th.bool),
            infos
        )
    
    def reset(self, seed=None, options=None):
        states, _ = super().reset(seed=seed, options=options)
        return to_tensor(states)