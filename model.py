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

# Step 13 - bandit_parameter_study
import numpy as np

def bandit_parameter_study(n_runs, n_steps, seed, settings):
    k = 10
    results = {}

    for setting in settings:
        method = setting['method']
        param = setting['param']
        nonstationary = setting.get('nonstationary', False)

        label = f"{method}({param})"
        if nonstationary:
            label += ",ns"

        final_rewards = np.zeros(n_runs)

        for run in range(n_runs):
            if nonstationary:
                true_values = np.zeros(k)
            else:
                true_values = create_bandit_testbed(k, seed + run)
            rng = np.random.default_rng(seed + run)

            q = np.zeros(k)
            counts = np.zeros(k, dtype=int)
            prefs = np.zeros(k)
            average_reward = 0.0
            if method == 'optimistic':
                q = optimistic_initialization(k, param)

            reward = 0.0
            for t in range(1, n_steps + 1):
                # 1. select an action
                if method == 'epsilon_greedy':
                    action = epsilon_greedy_action(q, param, rng)
                elif method == 'constant_step':
                    action = epsilon_greedy_action(q, 0.1, rng)
                elif method == 'optimistic':
                    action = epsilon_greedy_action(q, 0.0, rng)
                elif method == 'ucb':
                    action = ucb_action_select(q, counts, t, param)
                elif method == 'gradient':
                    e = np.exp(prefs - np.max(prefs))
                    pi = e / e.sum()
                    action = int(rng.choice(k, p=pi))
                else:
                    raise ValueError(f"Unknown method: {method}")

                # 2. pull the arm
                reward = pull_arm(true_values, action, rng)

                # 3. update the agent
                if method in ('epsilon_greedy', 'ucb'):
                    q, counts = sample_average_update(q, counts, action, reward)
                elif method == 'constant_step':
                    q = constant_step_size_update(q, action, reward, param)
                elif method == 'optimistic':
                    q = constant_step_size_update(q, action, reward, 0.1)
                else:  # gradient
                    average_reward += (reward - average_reward) / t
                    prefs = gradient_bandit_update(
                        prefs, action, reward, average_reward, param
                    )

                # 4. the world drifts
                if nonstationary:
                    true_values = apply_random_walk_drift(true_values, 0.01, rng)

            final_rewards[run] = reward

        results[label] = float(final_rewards.mean())

    return results

# Step 14 - build_gridworld_mdp
def build_gridworld_mdp():
    n_states = 16
    n_actions = 4
    size = 4
    terminals = {0, 15}
    # action -> (row change, column change): north, east, south, west
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    P = []
    for s in range(n_states):
        P.append([])
        for a in range(n_actions):
            if s in terminals:
                P[s].append([(1.0, s, 0.0)])
            else:
                row, col = divmod(s, size)
                dr, dc = deltas[a]
                new_row = min(max(row + dr, 0), size - 1)
                new_col = min(max(col + dc, 0), size - 1)
                next_s = new_row * size + new_col
                P[s].append([(1.0, next_s, -1.0)])

    return {'n_states': n_states, 'n_actions': n_actions, 'P': P}

# Step 15 - iterative_policy_evaluation
import numpy as np

def iterative_policy_evaluation(policy, mdp, gamma, theta):
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']

    policy = np.asarray(policy)
    if policy.ndim == 1:
        pi = np.zeros((n_states, n_actions))
        pi[np.arange(n_states), policy.astype(int)] = 1.0
    else:
        pi = policy

    V = np.zeros(n_states)

    while True:
        delta = 0.0
        for s in range(n_states):
            v = V[s]
            new_v = 0.0
            for a in range(n_actions):
                for prob, next_s, reward in P[s][a]:
                    new_v += pi[s, a] * prob * (reward + gamma * V[next_s])
            V[s] = new_v
            delta = max(delta, abs(v - new_v))
        if delta < theta:
            break

    return V

# Step 16 - greedy_policy_improvement
import numpy as np

def greedy_policy_improvement(state_values, mdp, gamma):
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        q = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_s, reward in P[s][a]:
                q[a] += prob * (reward + gamma * state_values[next_s])
        policy[s] = np.argmax(q)

    return policy

# Step 17 - policy_iteration
import numpy as np

def policy_iteration(mdp, gamma, theta):
    n_states = mdp['n_states']
    policy = np.zeros(n_states, dtype=int)

    while True:
        state_values = iterative_policy_evaluation(policy, mdp, gamma, theta)
        new_policy = greedy_policy_improvement(state_values, mdp, gamma)

        if np.array_equal(new_policy, policy):
            return state_values, new_policy

        policy = new_policy

# Step 18 - value_iteration (not yet solved)
# TODO: implement

# Step 19 - build_gambler_mdp (not yet solved)
# TODO: implement

# Step 20 - gambler_value_iteration (not yet solved)
# TODO: implement

# Step 21 - extract_optimal_stakes (not yet solved)
# TODO: implement

