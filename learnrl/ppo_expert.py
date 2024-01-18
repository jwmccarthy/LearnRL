import torch as th
import torch.nn as nn
import gymnasium as gym

from envs import SyncTorchEnv
from modules import AgentModule, CriticModule, DiscriminatorModule
from rollout import RolloutCollector
from algos import PPO
from utils import stack_states


ROLLOUT_SIZE = 4096


if __name__ == "__main__":
    env = SyncTorchEnv("ALE/Asteroids-v5", num_envs=3)

    # Training Expert

    agent_net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(env.state_dim[0] * env.state_dim[1] * 16, 64),
        nn.Tanh(),
        nn.Linear(64, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )
    agent_module = AgentModule(agent_net, env.single_action_space)

    critic_net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(env.state_dim[0] * env.state_dim[1] * 16, 64),
        nn.Tanh(),
        nn.Linear(64, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )
    critic_module = CriticModule(critic_net)

    collector = RolloutCollector(env, agent_module, ROLLOUT_SIZE)

    ppo = PPO(
        agent_module, 
        critic_module, 
        collector,
        ent_coef=0.1
    )

    ppo.learn(int(5e6))

    # Collect Expert Demos

    expert_collector = RolloutCollector(env, agent_module, 50000)
    buffer, _, _ = expert_collector.collect()

    # create demo pairs
    expert_demos = stack_states(buffer.states, buffer.next_states)
    th.save(expert_demos, "expert_demos.pt")

    # test env
    env = SyncTorchEnv("ALE/Asteroids-v5", render_mode="human")

    states = env.reset()
    for i in range(100000):
        action, _, _ = agent_module(states)
        states, _, term, trunc, _ = env.step(action)
        if term or trunc:
            states = env.reset()