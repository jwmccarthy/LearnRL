import numpy as np
import torch as th
from utils import to_tensor, TensorBuffer


class RolloutCollector:

    def __init__(
        self,
        env,
        agent,
        size
    ):
        self.env = env
        self.agent = agent

        # buffer dims
        self.size = size
        self.num_envs = self.env.num_envs
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

    def collect(self):
        # init buffer
        # TODO: Do this in such as way as to accommodate added tensors
        # Not efficient to create a new one every time
        buffer_dim = (self.size, self.num_envs)
        self.buffer = TensorBuffer.from_dims(
            states=buffer_dim + self.state_dim,
            next_states=buffer_dim + self.state_dim,
            actions=buffer_dim + self.action_dim,
            rewards=buffer_dim,
            logprobs=buffer_dim,
            terms=buffer_dim,
            truncs=buffer_dim,
            starts=buffer_dim
        )

        last_states = self.env.reset()
        curr_states = th.zeros_like(last_states)
        last_starts = th.zeros((self.num_envs), dtype=th.bool)

        # logging
        ep_rewards, ep_lengths = [], []
        ep_reward = np.zeros((self.num_envs))
        ep_length = np.zeros((self.num_envs))

        for t in range(self.size):            
            with th.no_grad():
                actions, logprobs, _ = self.agent(last_states)

            next_states, rewards, terms, truncs, infos = self.env.step(actions)

            # retain last state upon reset
            for i, (term, trunc) in enumerate(zip(terms, truncs)):

                # update episode stats
                ep_reward[i] += rewards[i]
                ep_length[i] += 1

                if term or trunc:
                    curr_states[i] = to_tensor(infos["final_observation"][i])

                    # log episode rewards, lengths
                    ep_rewards.append(ep_reward[i])
                    ep_lengths.append(ep_length[i])
                    ep_reward[i], ep_length[i] = 0, 0
                else:
                    curr_states[i] = next_states[i]

            # add all values to buffer
            self.buffer[t] = (
                last_states,
                curr_states,
                actions,
                rewards,
                logprobs,
                terms,
                truncs,
                last_starts
            )

            last_starts = terms | truncs
            last_states = next_states

        return self.buffer, np.mean(ep_rewards), np.mean(ep_lengths)