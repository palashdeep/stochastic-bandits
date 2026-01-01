import numpy as np

from experiments.plots import *
from policies import GreedyPolicy, e_GreedyPolicy, UCBPolicy, ThompsonSamplingPolicy
from environment import Environment

def run_experiment(env, policy, horizon):
    """Runs one experiment over horizon"""
    rewards = []
    regrets = []
    arms_pulled = []

    for _ in range(horizon):
        arm = policy.select_arm(env.arms)
        reward, regret = env.pull(arm)
        policy.update(arm, reward)

        rewards.append(reward)
        regrets.append(regret)
        arms_pulled.append(arm)

    return rewards, regrets, arms_pulled

def run_ts_experiment(env, ts_policy, horizon):
    """Runs one experiment with TS and saves alpha/beta history"""
    alpha_history = []
    beta_history = []

    for t in range(horizon):
        arm = ts_policy.select_arm(arms)
        reward, regret = env.pull(arm)
        ts_policy.update(arm, reward)

        alpha_history.append(ts_policy.alpha.copy())
        beta_history.append(ts_policy.beta.copy())

    return alpha_history, beta_history

"""
Main experiment runner.
Generates all figures reported in the README.
"""

if __name__ == "__main__":

    # Experiment 1: To compare different policies

    horizon = 5000
    runs = 1000
    arms = 3
    arms_arr = [1, 2, 3]

    env = Environment(name="Environment 1", arms=arms, mus=[0.8, 0.9, 0.7])
    policies = [
        GreedyPolicy(env.arms),
        e_GreedyPolicy(env.arms, epsilon=0.05),
        UCBPolicy(env.arms),
        ThompsonSamplingPolicy(env.arms)
    ]

    results = {}
    avg_results = {}

    for policy in policies:
        for run in range(runs):
            env.reset(env.name, env.arms, env.mus)
            policy.reset(policy.name, env.arms, policy.rules)
            rewards, regrets, arms_pulled = run_experiment(env, policy, horizon)
            if policy.name not in results:
                results[policy.name] = {
                    "rewards": [],
                    "regrets": [],
                    "arms": []
                }
            results[policy.name]["rewards"].append(rewards)
            results[policy.name]["regrets"].append(regrets)
            results[policy.name]["arms"].append(arms_pulled)

        avg_regrets = np.mean(results[policy.name]["regrets"], axis=0)
        std_regrets = np.std(results[policy.name]["regrets"], axis=0)
        arms_pulled = np.array(results[policy.name]["arms"])
        arms_pull = {
            arm: np.mean(arms_pulled == arm, axis=0)
            for arm in arms_arr
        }

        avg_results[policy.name] = (avg_regrets, std_regrets, arms_pull)

    plot_results(avg_results, horizon, True)
    plot_pulls(avg_results, horizon, True)
    plot_regret_distribution(results, True)

    # Experiment 2: for TS posterior evolution

    horizon = 1000
    
    env = Environment(name="Environment 2", arms=3, mus=[0.8, 0.9, 0.7])
    policy = ThompsonSamplingPolicy(env.arms)

    arm_to_plot = 2
    timesteps = [10, 50, 200, 1000]

    alpha_hist, beta_hist = run_ts_experiment(env, policy, horizon)

    plot_ts_posterior_evolution(alpha_hist, beta_hist, arm_to_plot-1, timesteps, True)
