import sys
import time
from collections import deque
import numpy as np

from constants import *
from environment import *
from state import State

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

"""


class Solver:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.states = []
        self.policy = {}  # state:action
        self.values = {}  # state:value
        self.terminal_states = []
        self.converged = False
        self.EPSILON = 0.01
        self.t_model = None
        self.r_model = None
        self.state_indices = None
        self.target_center_dict = self.get_target_center_dict()

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5, 6]

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        self.bfs_initialise()
        self.terminal_states = [s for s in self.states if self.environment.is_solved(s)]
        self.policy = {s: BEE_ACTIONS[0] for s in self.states}
        self.values = {s: 0 for s in self.states}
        self.converged = False

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.converged

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        new_policy = dict()
        max_diff = 0

        # Loop states
        for s in self.states:
            best_q = -float('inf')
            best_a = None

            # Loop over possible actions
            for a in BEE_ACTIONS:
                total = 0

                # Get transition outcomes
                for prob, next_state, reward in self.get_transition_outcomes(s, a):
                    total += prob * (reward + (self.environment.gamma * self.values.get(next_state)))

                # Record the best action
                if total > best_q:
                    best_q = total
                    best_a = a

            max_diff = max(max_diff, abs(self.values[s] - best_q))

            self.values[s] = best_q
            new_policy[s] = best_a

        if max_diff < self.EPSILON:
            self.converged = True

        self.policy = new_policy

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while True:
            self.vi_iteration()

            # NOTE: vi_iteration is always called before vi_is_converged
            if self.vi_is_converged():
                break

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        return self.values[state]

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.policy[state]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        self.bfs_initialise()
        self.terminal_states = {s for s in self.states if self.environment.is_solved(s)}
        print(self.terminal_states)
        self.state_indices = {s: i for i, s in enumerate(self.states)}

        self.t_model = np.zeros([len(self.states), len(BEE_ACTIONS), len(self.states)])
        for i, s in enumerate(self.states):
            for j, a in enumerate(BEE_ACTIONS):
                for prob, next_state, _ in self.get_transition_outcomes(s, a):
                    # only add this for ex9 as bfs could not find all states in one go
                    if next_state in self.state_indices:
                        k = self.state_indices[next_state]
                        self.t_model[i][j][k] += prob

        self.policy = np.zeros([len(self.states)], dtype=np.int64)
        r_model = np.zeros([len(self.states), len(BEE_ACTIONS)])



        for i, s in enumerate(self.states):
            for a in range(len(BEE_ACTIONS)):
                expected_reward = 0
                for prob, next_state, reward in self.get_transition_outcomes(s, BEE_ACTIONS[a]):
                    # only add this for ex9 as bfs could not find all states in one go
                    expected_reward += prob * reward
                    # if next_state in self.state_indices:
                    #     expected_reward_list = self.process_reward(prob, next_state, reward)
                    #     for ex_prob, _, ex_reward in expected_reward_list:
                    #         expected_reward += ex_prob * ex_reward
                r_model[i, a] = expected_reward

        self.r_model = r_model
        self.converged = False

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.converged

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        v_pi = self.policy_evaluation()
        self.policy_improvement(v_pi)

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while True:
            self.pi_iteration()
            # NOTE: pi_iteration is always called before pi_is_converged
            if self.pi_is_converged():
                break

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.policy[self.state_indices[state]]

    # === Helper Methods ===============================================================================================

    def get_successors(self, state):
        successors = []
        for action in BEE_ACTIONS:
            cost, next_state = self.environment.apply_dynamics(state, action)
            if next_state != state:
                successors.append(next_state)
        return successors

    def bfs_initialise(self):
        initial_state = self.environment.get_init_state()
        visited = {initial_state}
        frontier = deque([initial_state])
        states = set()

        while frontier:
            state = frontier.popleft()
            states.add(state)
            if self.environment.is_solved(state):
                continue

            for successor_state in self.get_successors(state):
                if successor_state not in visited:
                    visited.add(successor_state)
                    frontier.append(successor_state)

        self.states = list(states)

    def get_transition_outcomes(self, state, action):
        if state in self.terminal_states:
            return [(1.0, state, 0)]

        outcomes = []
        cw_double_prob = self.environment.double_move_probs[action] * self.environment.drift_cw_probs[action]
        ccw_double_prob = self.environment.double_move_probs[action] * self.environment.drift_ccw_probs[action]
        cw_prob = self.environment.drift_cw_probs[action] - cw_double_prob
        ccw_prob = self.environment.drift_ccw_probs[action] - ccw_double_prob
        double_prob = self.environment.double_move_probs[action] - cw_double_prob - ccw_double_prob
        direct_prob = 1 - cw_prob - ccw_prob - double_prob - cw_double_prob - ccw_double_prob
        probs = [direct_prob, cw_prob, ccw_prob, double_prob, cw_double_prob, ccw_double_prob]
        actions = [[action], [SPIN_RIGHT, action], [SPIN_LEFT, action], [action, action],
                   [SPIN_RIGHT, action, action], [SPIN_LEFT, action, action]]

        for movement, prob in zip(actions, probs):
            next_state = state
            total_reward = float('inf')
            for act in movement:
                reward, next_state = self.environment.apply_dynamics(next_state, act)
                if reward < total_reward:
                    total_reward = reward
            else:
                outcomes.append((prob, next_state, total_reward))
        return outcomes

    def process_reward(self, prob, next_state, reward):
        outcomes = []
        # if next_state.is_on_edge():
        #     reward -= 0.5
        # if next_state.is_next_to_obstacle():
        #     reward -= 0.5
        # if next_state.is_next_to_thorn():
        #     reward -= 1.5  # 3 times the penalty of collision
        # if next_state.is_not_adjacent_widget():
        #     reward -= next_state.distance_to_widget()
        #
        center_dict = self.target_center_dict
        widget_dict = dict(zip(next_state.widget_centres, next_state.environment.widget_types))

        for widget_location, widget_type in widget_dict.items():
            for center, center_type in center_dict.items():
                if center_type == widget_type:
                    x_distance = center[0] - widget_location[0]
                    y_distance = center[1] - widget_location[1]
                    distance = (abs(x_distance) + abs(x_distance + y_distance) + abs(y_distance)) / 2
                    reward -= distance
        outcomes.append((prob, next_state, reward))
        return outcomes

    def policy_evaluation(self):
        """
        Evaluate the current policy.
        """
        # use linear algebra for policy evaluation
        # V^pi = R + gamma T^pi V^pi
        # (I - gamma * T^pi) V^pi = R
        # Ax = b; A = (I - gamma * T^pi),  b = R

        # indices of every state
        state_indices_arr = np.array(range(len(self.states)))
        # index into t_model to select only entries where a = pi(s)
        t_pi = self.t_model[state_indices_arr, self.policy]
        r_pi = self.r_model[state_indices_arr, self.policy]
        # solve for V^pi(s) using linear algebra
        v_pi = np.linalg.solve(np.identity(len(self.states)) - self.environment.gamma * t_pi, r_pi)
        # convert values vector to dict and return
        return {s: v_pi[self.state_indices[s]] for s in self.states}

    def policy_improvement(self, v_pi):
        """
        Improve the current policy.
        """
        policy_changed = False
        print(v_pi)
        for s in self.states:
            best_q = -float('inf')
            best_a = None
            for a in range(len(BEE_ACTIONS)):
                total = 0
                for prob, next_state, reward in self.get_transition_outcomes(s, a):
                    # only add this for ex9 as bfs could not find all states in one go
                    total += prob * (reward + self.environment.gamma * v_pi[next_state])
                    # if next_state in self.state_indices:
                    #     expected_reward_list = self.process_reward(prob, next_state, reward)
                    #     for ex_prob, ex_next_state, ex_reward in expected_reward_list:
                    #         total += ex_prob * (ex_reward + self.environment.gamma * v_pi[ex_next_state])

                if total > best_q:
                    best_q = total
                    best_a = a

            # update state action with best action
            if self.policy[self.state_indices[s]] != best_a:
                policy_changed = True
                self.policy[self.state_indices[s]] = best_a

        self.converged = not policy_changed

    def get_target_center_dict(self):
        """Find the center points of the target widgets, reuse the code from my assignment 1"""
        target_list_copy = self.environment.target_list.copy()
        center_points = {}
        nums = sorted((int(num) for num in self.environment.widget_types), reverse=True)
        for n in nums:
            target_found = False
            for target in target_list_copy.copy():
                row, col = target
                neighbors = get_all_adjacent_cell_coords(row, col)
                neighbor_count = sum(1 for neighbor in neighbors if neighbor in target_list_copy)
                if neighbor_count >= n - 1:
                    center_points[target] = str(n)
                    target_list_copy.remove(target)
                    remove_neighbors_count = 0
                    for neighbor in neighbors:
                        if neighbor in target_list_copy:
                            target_list_copy.remove(neighbor)
                            remove_neighbors_count += 1
                        if remove_neighbors_count >= n - 1:
                            break
                    target_found = True
                    break
            if target_found:
                continue

        while len(center_points) < self.environment.n_widgets and target_list_copy:
            random_point = random.choice(target_list_copy)
            center_points[random_point] = len(target_list_copy)
            target_list_copy.remove(random_point)

        return center_points


def get_all_adjacent_cell_coords(row, col):
    """
    Get the coordinates of all adjacent cells in a hexagonal grid.
    :param cell: Tuple (row, col) representing the current cell
    :return: List of tuples representing the coordinates of all adjacent cells
    """
    if col % 2 == 0:  # even column
        adjacent_cells = [
            (row - 1, col),  # UP
            (row + 1, col),  # DOWN
            (row - 1, col - 1),  # UP_LEFT
            (row - 1, col + 1),  # UP_RIGHT
            (row, col - 1),  # DOWN_LEFT
            (row, col + 1)  # DOWN_RIGHT
        ]
    else:  # odd column
        adjacent_cells = [
            (row - 1, col),  # UP
            (row + 1, col),  # DOWN
            (row, col - 1),  # UP_LEFT
            (row, col + 1),  # UP_RIGHT
            (row + 1, col - 1),  # DOWN_LEFT
            (row + 1, col + 1)  # DOWN_RIGHT
        ]
    return adjacent_cells
