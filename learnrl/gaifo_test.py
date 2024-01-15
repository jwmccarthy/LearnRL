import torch.nn as nn
import gymnasium as gym

from envs import SyncTorchEnv
from modules import AgentModule, CriticModule, DiscriminatorModule
from rollout import RolloutCollector
from algos import PPO, GAIFO
from utils import stack_states
from copy import deepcopy


ROLLOUT_SIZE = 2048


if __name__ == "__main__":
    env = SyncTorchEnv("LunarLander-v2")

    # Training Expert

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

    discrim_net = nn.Sequential(
        nn.Linear(env.flat_dim*2, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    discrim_module = DiscriminatorModule(discrim_net)

    collector = RolloutCollector(env, agent_module, ROLLOUT_SIZE)

    ppo = PPO(
        agent_module, 
        critic_module, 
        collector
    )

    ppo.learn(int(1e6))

    # Collect Expert Demos

    expert_collector = RolloutCollector(env, agent_module, 50000)
    expert_collector.collect()

    expert_demos = stack_states(
        expert_collector.buffer.states,
        expert_collector.buffer.next_states
    )

    # Train w/ Expert Demos

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

    discrim_net = nn.Sequential(
        nn.Linear(env.flat_dim*2, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    discrim_module = DiscriminatorModule(discrim_net)

    collector = RolloutCollector(env, agent_module, ROLLOUT_SIZE)

    gaifo = GAIFO(
        agent_module, 
        critic_module,
        DiscriminatorModule(discrim_net),
        collector,
        expert_demos
    )

    gaifo.learn(int(1e6))

    # test env
    env = SyncTorchEnv("LunarLander-v2", render_mode="human")

    states = env.reset()
    for i in range(100000):
        action, _, _ = agent_module(states)
        states, _, term, trunc, _ = env.step(action)
        if term or trunc:
            states = env.reset()