import sys
from world.environment import Environment
from agents.base_agent import BaseAgent
import numpy as np

from world.helpers import action_to_direction
from world.rewards.weighted_reward import get_weighted_reward_fn


class MCOnPolicyAgent(BaseAgent):
    """
    Monte Carlo agent using on-policy control.
    """

    DEFAULT_REWARD_FN = get_weighted_reward_fn(100, 10000)
    DEFAULT_EPSILON = 0.4
    DEFAULT_ITER_SCALER = 20.0

    def __init__(
        self,
        env: Environment,
        iter_scaler: float = DEFAULT_ITER_SCALER,
        epsilon: float = DEFAULT_EPSILON,
        convergence_threshold: float = 0.001,
        reward_fn: callable = DEFAULT_REWARD_FN,
    ):
        """
        Args:
            `env`: The environment the agent is acting in.
            `iter_scaler`: Determines the maximum number of iterations
                in every episode, relative to the size of the grid.
            `epsilon`: The epsilon value for the epsilon-soft policy.
            `convergence_threshold`: The upper bound of ratio change in
                the q-values required for convergence.
            `reward_fn`: The reward function to use for evaluating
                moves in the simulated episodes.
        """
        super(MCOnPolicyAgent, self).__init__()

        self.env = env
        self.iters = int(env.grid.shape[0] * env.grid.shape[1] * iter_scaler)

        assert 0 < epsilon <= 1
        self.soft_zero = epsilon / self.NUM_ACTIONS
        self.soft_one = 1 - epsilon + self.soft_zero

        self.converged = False
        self.convergence_threshold = convergence_threshold

        self.reward_fn = reward_fn

        # On-policy control specific initialization
        self.q_table = np.zeros((*env.grid.shape, self.NUM_ACTIONS))
        self.return_sums = np.zeros((*env.grid.shape, self.NUM_ACTIONS))
        self.return_counts = np.zeros((*env.grid.shape, self.NUM_ACTIONS))
        # Initialize policy with epsilon = 1.0 (i.e. uniform random policy)
        self.policy = np.ones((*env.grid.shape, self.NUM_ACTIONS)) * (
            1.0 / self.NUM_ACTIONS
        )

        # train agent
        self.run_simulations(env.agent_pos)

    def generate_episode(
        self, start_state: tuple[int, int], policy: np.ndarray
    ) -> list[tuple[tuple[int, int], int, float]]:
        """
        Generate episode starting at specified state.

        Args:
            `start_state`: The starting state of the episode.
            `policy`: The policy to follow.
        Returns:
            `list[tuple[tuple[int, int], int, float]]`: The episode, consisting of
                tuples of state, action, and return.
        """
        episode = []
        state = start_state

        for _ in range(self.iters):
            action = np.random.choice(self.ACTIONS, p=policy[*state, :])
            new_state, reward, terminated, actual_action = self.simulate_step(
                state, action, self.reward_fn
            )
            episode.append([state, actual_action, reward])
            state = new_state

            if terminated:
                break

        # Calculate returns after each state-action pair,
        # by iterating backwards through the episode
        total_return = 0
        for i, (state, action, reward) in enumerate(episode[::-1]):
            total_return += reward
            episode[-i - 1] = (state, action, total_return)

        return episode

    def run_simulations(self, start_state: tuple[int, int]):
        """
        Update the agent based on simulations starting in the current state.
        If the policy had previously converged, the policy is not updated.

        Args:
            `state`: The current state.
        """

        while not self.converged:
            episode = self.generate_episode(start_state, self.policy)

            self.converged = True
            G = dict()
            seen_states = set()
            max_change = 0

            for state, action, return_after in episode:
                if (state, action) not in G:
                    G[(state, action)] = return_after
                    seen_states.add(state)

                self.return_sums[*state, action] += G[(state, action)]
                self.return_counts[*state, action] += 1

                original_q = self.q_table[*state, action]
                self.q_table[*state, action] = (
                    self.return_sums[*state, action]
                    / self.return_counts[*state, action]
                )
                change = abs(self.q_table[*state, action] - original_q) / abs(
                    original_q + 1e-6
                )
                if change > self.convergence_threshold:
                    self.converged = False
                max_change = max(change, max_change)

            print(f"\033[K{max_change}", end="\r")
            sys.stdout.flush()

            for state in seen_states:
                greedy_action = np.argmax(self.q_table[*state, :])
                self.policy[*state, :] = self.soft_zero
                self.policy[*state, greedy_action] = self.soft_one

    def take_action(self, state: tuple[int, int]) -> int:
        """
        Take an action based on the current policy.

        Args:
            `state`: The current state.
        Returns:
            `int`: The action to take.
        """

        return np.random.choice(self.ACTIONS, p=self.policy[*state, :])

    def update(self, _state: tuple[int, int], _reward: float, _action: int):
        pass
