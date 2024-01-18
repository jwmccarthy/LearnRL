import torch as th
import torch.nn as nn
from envs import SyncTorchEnv
from modules import AgentModule, CriticModule, DiscriminatorModule
from rollout import RolloutCollector
from algos import GAIFO


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

    expert_demos = th.load("expert_demos.pt")

    gaifo = GAIFO(
        agent_module, 
        critic_module,
        DiscriminatorModule(discrim_net),
        collector,
        expert_demos,
        ent_coef=0.1
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