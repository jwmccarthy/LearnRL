import attrs
import numpy as np
import torch as th


TENSOR_FIELD = attrs.field(converter=th.as_tensor)


@attrs.define()
class RolloutSample:
    states: th.Tensor = TENSOR_FIELD
    actions: th.Tensor = TENSOR_FIELD
    values: th.Tensor = TENSOR_FIELD
    logprobs: th.Tensor = TENSOR_FIELD
    returns: th.Tensor = TENSOR_FIELD
    advantages: th.Tensor = TENSOR_FIELD


class RolloutBuffer:

    states: np.ndarray
    actions: np.ndarray
    starts: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    logprobs: np.ndarray
    
    # derived values
    returns: np.ndarray
    advantages: np.ndarray

    def __init__(self, size, num_envs, state_dim, action_dim):
        # data dimensions
        self.size = size
        self.num_envs = num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim

        self._init_buffers()

    def _init_buffers(self):
        self.states = np.zeros((self.size, self.num_envs, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.size, self.num_envs, self.action_dim), dtype=np.float32)
        self.starts = np.zeros((self.size, self.num_envs), dtype=np.float32)
        self.rewards = np.zeros((self.size, self.num_envs), dtyp=np.float32)
        self.values = np.zeros((self.size, self.num_envs), dtype=np.float32)
        self.logprobs = np.zeros((self.size, self.num_envs), dtype=np.float32)

    def __getitem__(self, idx):
        return RolloutSample(
            self.states[idx],
            self.actions[idx],
            self.values[idx],
            self.logprobs[idx],
            self.returns[idx],
            self.advantages[idx]
        )
    
    def __setitem__(self, idx, experience):
        (
            self.states[idx],
            self.actions[idx],
            self.starts[idx],
            self.rewards[idx],
            self.values[idx],
            self.logprobs[idx]
        ) = experience


class RolloutCollector:

    def __init__(
        self,
        env,
        agent,
        size,
        gae_gamma=0.99,
        gae_lambda=0.95
    ):
        self.env = env
        self.agent = agent

        # data dimensions
        self.size = size
        self.num_envs = env.num_envs

        # advantage estimation hyperparameters
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda

        # may change w/ TODO #1
        self.state_dim = self.agent.state_dim
        self.action_dim = self.agent.action_dim

        self.buffer = RolloutBuffer(
            self.size, self.num_envs, self.state_dim, self.action_dim
        )

    def collect_rollout(self):
        last_starts = np.zeros((self.num_envs), dtype=np.float32)
        last_states = self.env.reset()[0]

        for t in range(self.size):

            with th.no_grad():
                states_tensor = th.as_tensor(last_states)
                actions, values, logprobs, _ = self.agent(states_tensor)
            actions = actions.cpu().numpy()

            # gather experience from env
            next_states, rewards, terms, truncs, infos = self.env.step(actions)

            # bootstrap in case of timeout
            for i, term, trunc in enumerate(zip(terms, truncs)):
                if term and not trunc:
                    term_state = th.as_tensor(infos["terminal_observation"][i])
                    term_value = self.agent.get_value(term_state)
                    rewards[i] += self.gae_gamma * term_value

            self.buffer[t] = (
                last_states,
                actions,
                rewards,
                last_starts,
                values.cpu().numpy(),
                logprobs.cpu().numpy()
            )

            last_starts = terms | truncs
            last_states = next_states

    def calc_advantages(self):
        pass