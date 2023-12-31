import torch.nn as nn
import gymnasium as gym

from envs import SyncTorchEnv
from modules import AgentModule, CriticModule
from rollout import RolloutCollector


ROLLOUT_SIZE = 2048


if __name__ == "__main__":
    env = SyncTorchEnv(3 * [lambda: gym.make("LunarLander-v2")])

    agent_net = nn.Sequential(
        nn.Linear(env.flat_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, env.logit_dim)
    )
    agent_module = AgentModule(agent_net, env.single_action_space)

    critic_net = nn.Sequential(
        nn.Linear(env.flat_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    critic_module = CriticModule(critic_net)

    collector = RolloutCollector(env, agent_module, ROLLOUT_SIZE)

    print(collector.buffer.flatten())
    print(collector.buffer["states"])
    print(collector.buffer[0].get("actions", "states"))