import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as opt
import torch.functional as F
from torch.distributions import Categorical

import gymnasium as gym
from tqdm import tqdm
from stable_baselines3 import PPO

from buffers import RolloutBuffer


# --- helper functions ---

def dims_to_sequential(in_dim, hidden_dims, out_dim, act_func):
    network = nn.Sequential()
    last_dim = in_dim
    for next_dim in hidden_dims:
        network.append(nn.Linear(last_dim, next_dim))
        network.append(act_func())
        last_dim = next_dim
    network.append(nn.Linear(last_dim, out_dim))
    return network

def th_normalize(x):
    return (x - x.mean()) / (x.std() + 1e-10)


# --- initialize policy ---

# [X] policy distribution

def dist_from_action_space(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        return CategoricalDist(action_space.n)
    else:
        raise NotImplementedError(
            f"No distribution for action space of type {type(action_space)}"
        )

class CategoricalDist(nn.Module):

    def __init__(self, act_dim):
        super(CategoricalDist, self).__init__()

        self.act_dim = act_dim

    def sample(self):
        return self.dist.sample()
    
    def select(self):
        return self.dist.mode()
    
    def entropy(self):
        return self.dist.entropy()
    
    def log_prob(self, actions):
        return self.dist.log_prob(actions)
    
    def get_actions(self):
        # deterministic in eval mode
        if self.training:
            return self.sample()
        else:
            return self.select()

    def forward(self, logits, actions=None):
        self.dist = Categorical(logits=logits)

        # evaluate given actions
        if actions is None:
            actions = self.get_actions()

        log_probs = self.log_prob(actions)
        entropy = self.entropy()
        
        return actions, log_probs, entropy


# [X] agent class

class StochasticAgent(nn.Module):

    def __init__(
        self, 
        env,
        act_func=nn.ReLU,
        actor_arch=[64, 64],
        critic_arch=[64, 64],
        optimizer=opt.Adam,
        opt_kwargs={}
    ):
        super(StochasticAgent, self).__init__()

        # get action distribution based on action space
        self.act_dist = dist_from_action_space(env.single_action_space)

        # input, output dimensions
        self.obs_dim = env.single_observation_space.shape[0]
        self.act_dim = self.act_dist.act_dim

        # create actor network
        self.actor = dims_to_sequential(
            self.obs_dim, actor_arch, self.act_dim, act_func
        )

        # create critic network
        self.critic = dims_to_sequential(
            self.obs_dim, critic_arch, 1, act_func
        )

        self.optimizer = optimizer(self.parameters(), **opt_kwargs)

    def get_value(self, obs):
        obs_tensor = th.as_tensor(obs)
        return self.critic(obs_tensor)
    
    def forward(self, obs, actions=None):
        value = self.critic(obs)
        logits = self.actor(obs)
        actions, log_probs, entropy = self.act_dist(logits)
        return actions, value, log_probs, entropy


# --- initialize discriminator ---

# [X] discriminator class

class Discriminator(nn.Module):

    def __init__(
        self,
        env,
        act_func=nn.ReLU,
        arch=[64, 64],
        optimizer=opt.Adam,
        opt_kwargs={}
    ):
        super(Discriminator, self).__init__()

        # initialize feedforward discriminator
        self.net = dims_to_sequential(2* env.single_observation_space.shape[0],
                                      arch, 1, act_func)
        self.net.append(nn.Sigmoid())
        
        self.optim = optimizer(self.parameters(), **opt_kwargs)

    def forward(self, state_tr):
        return self.net(state_tr)


# --- implement PPO ---

# load/obtain expert demonstrations

def save_ppo_expert_demos():

    train_timesteps = int(5e5)
    expert_timesteps = int(5e5)

    # initialize vector env
    env = gym.make("LunarLander-v2")

    # train model
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(train_timesteps, progress_bar=True)

    # save model
    model.save("ppo_lunar")

    # save expert demonstrations
    states = np.zeros((expert_timesteps, env.observation_space.shape[0]))

    obs = env.reset()[0]
    for t in tqdm(range(expert_timesteps)):
        states[t] = obs
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            obs = env.reset()[0]

    with open("demos/state_tr.npy", "wb") as f:
        np.save(f, states)

    return states

def load_ppo_expert_demos():
    demos = np.load("demos/state_tr.npy")
    return th.as_tensor(demos, dtype=th.float32)

def create_state_transitions(states):
    pairs = th.stack((states[:-1], states[1:]), dim=states.ndim-1)
    return pairs.flatten(start_dim=pairs.ndim-1)

def sample_state_transitions(pairs, n=1):
    rand_idx = th.randperm(len(pairs))
    for batch_start in range(0, BUFFER_SIZE * n, BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch_idx = rand_idx[batch_start:batch_end]
        yield pairs[batch_idx]

expert_states = load_ppo_expert_demos()
expert_states = create_state_transitions(expert_states)


# --- initialize ---

# constants
# DEVICE = "cuda"
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPOCHS = 10
BATCH_SIZE = 64
BUFFER_SIZE = 2048
TOTAL_TIMESTEPS = int(5e5)

# env
num_envs = 3
vec_env = gym.vector.SyncVectorEnv(
    num_envs * [lambda: gym.make("LunarLander-v2")]
)

# agent
agent = StochasticAgent(vec_env)

# discriminator
discriminator = Discriminator(vec_env)

# rollout buffer
buffer = RolloutBuffer(
    BUFFER_SIZE, agent.obs_dim, agent.act_dim, num_envs=num_envs,
    gamma=GAMMA, gae_lambda=GAE_LAMBDA
)


# --- collect rollout ---

def collect_rollout():
    last_starts = np.zeros((num_envs))
    last_obs, _ = vec_env.reset()

    for t in range(BUFFER_SIZE):

        final_obs = np.zeros((num_envs, vec_env.single_observation_space.shape[0]))

        with th.no_grad():
            obs_tensor = th.as_tensor(last_obs)
            actions, values, logprobs, _ = agent(obs_tensor)
        actions = actions.cpu().numpy()

        next_obs, rewards, terms, truncs, infos = vec_env.step(actions)
        next_starts = np.logical_or(terms, truncs)

        # bootstrap on episode termination
        for i in range(len(terms)):
            if terms[i] and not truncs[i]:
                final_obs[i] = infos["final_observation"][i]
        
        buffer.add(
            last_obs,
            final_obs,
            actions,
            rewards,
            last_starts,
            values,
            logprobs
        )
    
        last_starts = next_starts
        last_obs = next_obs

    # bootstrap in event of timeout
    with th.no_grad():
        obs_tensor = th.as_tensor(last_obs)
        last_values = agent.get_value(obs_tensor)
    buffer.compute_advantages(last_values, last_starts)


# --- train ---

d_optim = opt.Adam(discriminator.parameters())
p_optim = opt.Adam(agent.parameters())

for t in range(TOTAL_TIMESTEPS):

    collect_rollout()

    print(type(buffer.observations))
    agent_states = create_state_transitions(buffer.observations)

    # discriminator loss
    d_loss = nn.BCELoss()

    for e in range(EPOCHS):
        agent_states = th.as_tensor(agent_states)
        for a_data, e_data in zip(sample_state_transitions(agent_states), 
                                  sample_state_transitions(expert_states, n=3)):
            print(e_data.shape, a_data.shape)

            d_optim.zero_grad()

            # train w/ expert data
            e_pred = discriminator(e_data)
            e_loss = d_loss(e_pred, th.ones_like(e_pred))

            # train w/ agent data
            a_pred = discriminator(a_data)
            a_loss = d_loss(a_pred, th.zeros_like(a_pred))

            # backpropagate
            total_loss = e_loss + a_loss
            total_loss.backward()
            d_optim.step()

    # set reward according to discriminator outputs
    print(agent_states.shape)
    buffer.reward = -th.log(discriminator(agent_states))
    for i in range(buffer.buffer_size):
        for env in range(num_envs):
            term_obs = buffer.final_obs[i, env]
            if not np.any(term_obs):
                buffer.reward[i] += GAMMA * agent.get_value(term_obs)

    break

    # update policy via PPO updates with reward as first term in cross-entropy
    