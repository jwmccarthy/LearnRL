import numpy as np
import gymnasium as gym
from ppo import PPO
from tqdm import tqdm

# Create environment
env = gym.make("LunarLander-v2", render_mode="rgb_array", enable_wind=False)

# Instantiate the agent
model = PPO(env)

# Train the agent and display a progress bar
model.learn(total_timesteps=int(1e6))

N = int(1e6)

actions = np.zeros((N, ))

# env = gym.make("LunarLander-v2", render_mode="human", enable_wind=False)

# store expert values
expert_path = "demos/"

actions = np.zeros(N, dtype=np.int8)
observations = np.zeros((N, env.observation_space.shape[0]), dtype=float)

obs = env.reset()[0]

for i in tqdm(range(N)):
    action = model.predict(obs)

    actions[i] = action
    observations[i] = obs

    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        ep_t = 0
        obs = env.reset()[0]

np.save(expert_path+"act.npy", actions, allow_pickle=False)
np.save(expert_path+"obs.npy", observations, allow_pickle=False)