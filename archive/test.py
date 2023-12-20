import gymnasium as gym
from ppo import PPO

# Create environment
env = gym.make("LunarLander-v2", render_mode="rgb_array", enable_wind=False)

# Instantiate the agent
model = PPO(env)

# Train the agent and display a progress bar
model.learn(total_timesteps=int(1e6))

# # Enjoy trained agent
# vec_env = model.get_env()
vec_env = gym.make("LunarLander-v2", render_mode="human", enable_wind=False)
obs = vec_env.reset()[0]
for i in range(10000):
    action = model.predict(obs)
    obs, _, term, trunc, _ = vec_env.step(action)
    if term or trunc:
        ep_t = 0
        obs = vec_env.reset()[0]
    # vec_env.render("human")
