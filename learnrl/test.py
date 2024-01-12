import torch.nn as nn
import gymnasium as gym

from envs import SyncTorchEnv
from modules import AgentModule, CriticModule
from rollout import RolloutCollector
from algos import PPO


ROLLOUT_SIZE = 2048


if __name__ == "__main__":
    env = SyncTorchEnv("LunarLander-v2")

    agent_net = nn.Sequential(
        nn.Linear(env.flat_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, env.logit_dim)
    )
    agent_module = AgentModule(agent_net, env.single_action_space)

    critic_net = nn.Sequential(
        nn.Linear(env.flat_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )
    critic_module = CriticModule(critic_net)

    collector = RolloutCollector(env, agent_module, ROLLOUT_SIZE)

    ppo = PPO(agent_module, critic_module, collector)

    ppo.learn(int(1e6))

    # test env
    env = SyncTorchEnv("LunarLander-v2", render_mode="human")

    state = env.reset()
    for i in range(10000):
        action, _, _ = agent_module(state)
        state, _, term, trunc, _ = env.step(action)
        if term or trunc:
            state = env.reset()