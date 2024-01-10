import torch.nn as nn
import gymnasium as gym

from envs import SyncTorchEnv
from modules import AgentModule, CriticModule
from rollout import RolloutCollector
from algos import PPO


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
    buffer = collector.collect()

    print(buffer.flatten())
    print(buffer["states"])
    print(buffer[0].get("actions", "states"))

    for b in reversed(buffer):
        print(b)
        break

    ppo = PPO(agent_module, critic_module, collector)

    ppo.learn(1)

    print(buffer[1])