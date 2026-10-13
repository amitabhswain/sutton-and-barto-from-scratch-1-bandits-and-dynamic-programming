"""
Sutton and Barto from Scratch 1: Bandits and Dynamic Programming

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - create_bandit_testbed
import numpy as np

def create_bandit_testbed(k, seed, mean=0.0, std=1.0):
    rng = np.random.RandomState(seed)
    q_star = rng.normal(loc=mean, scale=std, size=k)
    return q_star

# Step 2 - pull_arm
import numpy as np

def pull_arm(true_values, action, rng):
    """Pull one arm and return reward = true value + unit-normal noise.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        action (int): Index of the arm to pull.
        rng (np.random.Generator): Seeded random generator for the noise.

    Returns:
        float: Stochastic reward for this pull.
    """
    noise = rng.normal()
    reward = true_values[action] + noise
    return float(reward)

# Step 3 - sample_average_update
import numpy as np

def sample_average_update(q_values, action_counts, action, reward):
    q_new = np.array(q_values, dtype=float)
    counts_new = np.array(action_counts, dtype=int)

    counts_new[action] += 1
    n = counts_new[action]
    q_new[action] += (reward - q_new[action]) / n

    return q_new, counts_new

# Step 4 - epsilon_greedy_action
import numpy as np

def epsilon_greedy_action(q_values, epsilon, rng):
    k = len(q_values)

    if rng.random() < epsilon:
        return int(rng.integers(0, k))

    return int(np.argmax(q_values))

# Step 5 - run_bandit_episode
import numpy as np

def run_bandit_episode(true_values, n_steps, epsilon, rng):
    """Run one bandit episode with epsilon-greedy selection and sample-average updates.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, actions) with shapes (n_steps,) and (n_steps,) of ints.
    """
    k = len(true_values)
    q_values = np.zeros(k)
    action_counts = np.zeros(k, dtype=int)
    rewards = np.zeros(n_steps)
    actions = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        action = epsilon_greedy_action(q_values, epsilon, rng)
        reward = pull_arm(true_values, action, rng)

        rewards[t] = reward
        actions[t] = action

        q_values, action_counts = sample_average_update(
            q_values, action_counts, action, reward
        )

    return rewards, actions

# Step 6 - track_rewards_and_optimal_actions
import numpy as np

def track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng):
    """Run one episode tracking rewards and optimal-arm choices.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, optimal_flags) each shape (n_steps,).
            optimal_flags entries are 0.0 or 1.0 floats.
    """
    rewards, actions = run_bandit_episode(true_values, n_steps, epsilon, rng)

    optimal_action = np.argmax(true_values)
    optimal_flags = (actions == optimal_action).astype(float)

    return rewards, optimal_flags

# Step 7 - average_bandit_curves
import numpy as np

def average_bandit_curves(k, n_runs, n_steps, epsilon, seed):
    all_rewards = np.zeros((n_runs, n_steps))
    all_optimal = np.zeros((n_runs, n_steps))

    for i in range(n_runs):
        true_values = create_bandit_testbed(k, seed + i)
        rng = np.random.default_rng(seed + i)
        rewards, optimal_flags = track_rewards_and_optimal_actions(
            true_values, n_steps, epsilon, rng
        )
        all_rewards[i] = rewards
        all_optimal[i] = optimal_flags

    mean_rewards = all_rewards.mean(axis=0)
    mean_optimal = all_optimal.mean(axis=0)
    return mean_rewards, mean_optimal

# Step 8 - apply_random_walk_drift
import numpy as np

def apply_random_walk_drift(true_values, drift_std, rng):
    noise = rng.normal(loc=0.0, scale=drift_std, size=true_values.shape)
    return true_values + noise

# Step 9 - constant_step_size_update
import numpy as np

def constant_step_size_update(q_values, action, reward, alpha):
    q_new = np.array(q_values, dtype=float)
    q_new[action] += alpha * (reward - q_new[action])
    return q_new

# Step 10 - optimistic_initialization
import numpy as np

def optimistic_initialization(k, initial_value):
    return np.full(k, initial_value, dtype=float)

# Step 11 - ucb_action_select
import numpy as np

def ucb_action_select(q_values, action_counts, timestep, c):
    """Select an action by upper-confidence-bound scores.

    Args:
        q_values (np.ndarray): Action-value estimates, shape (k,).
        action_counts (np.ndarray): Visit counts per action, shape (k,).
        timestep (int): Current time step t (>= 1).
        c (float): Exploration constant.

    Returns:
        int: Index of the selected action.
    """
    q = np.asarray(q_values, dtype=float)
    n = np.asarray(action_counts)

    scores = np.full(len(q), np.inf)
    visited = n > 0
    scores[visited] = q[visited] + c * np.sqrt(np.log(timestep) / n[visited])

    return int(np.argmax(scores))

# Step 12 - gradient_bandit_update
import numpy as np

def gradient_bandit_update(preferences, action, reward, average_reward, alpha):
    H = np.array(preferences, dtype=float)

    e = np.exp(H - np.max(H))
    pi = e / e.sum()

    advantage = reward - average_reward

    H -= alpha * advantage * pi
    H[action] += alpha * advantage

    return H

# Step 13 - bandit_parameter_study (not yet solved)
# TODO: implement

# Step 14 - build_gridworld_mdp (not yet solved)
# TODO: implement

# Step 15 - iterative_policy_evaluation (not yet solved)
# TODO: implement

# Step 16 - greedy_policy_improvement (not yet solved)
# TODO: implement

# Step 17 - policy_iteration (not yet solved)
# TODO: implement

# Step 18 - value_iteration (not yet solved)
# TODO: implement

# Step 19 - build_gambler_mdp (not yet solved)
# TODO: implement

# Step 20 - gambler_value_iteration (not yet solved)
# TODO: implement

# Step 21 - extract_optimal_stakes (not yet solved)
# TODO: implement

