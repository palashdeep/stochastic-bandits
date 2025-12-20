import matplotlib.pyplot as plt
import numpy as np
from policies import GreedyPolicy, e_GreedyPolicy, UCBPolicy, ThompsonSamplingPolicy
from environment import Environment

def run_experiment(env, policy, horizon):
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

def set_integer_xticks(ax, horizon, max_ticks=8):
    """
    Set integer x-ticks for a discrete horizon without clutter.
    """
    step = max(1, horizon // max_ticks)
    ticks = np.arange(0, horizon, step)

    ax.set_xticks(ticks)
    ax.set_xlim(0, horizon-1)

def plot_results(results, horizon=20):
    x = np.arange(horizon)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axes

    for policy_name, (regrets, std_regrets, _) in results.items():
        T = len(regrets)
        x = np.arange(T)

        # Cumulative mean regret
        cumulative_regrets = np.cumsum(regrets)
        cumulative_std = np.sqrt(np.cumsum(std_regrets ** 2))

        # ---- Plot 1: cumulative regret ----
        ax1.plot(x, cumulative_regrets, label=policy_name)

        # ---- Plot 2: cumulative regret with std bands ----
        line, = ax2.plot(x, cumulative_regrets, label=policy_name)
        ax2.fill_between(
            x,
            cumulative_regrets - cumulative_std,
            cumulative_regrets + cumulative_std,
            alpha=0.2,
            color=line.get_color()
        )

    ax1.set_title('Cumulative Regret')
    ax1.set_xlabel('Horizon')
    ax1.set_ylabel('Cumulative Regret')
    set_integer_xticks(ax1, horizon)
    ax1.legend()

    ax2.set_title('Cumulative Regret with ±1 Std Dev')
    ax2.set_xlabel('Horizon')
    ax2.set_ylabel('Cumulative Regret')
    set_integer_xticks(ax2, horizon)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_pulls(results, horizon=20):
    x = np.arange(horizon)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (policy_name, (_, _, arms_pull)) in zip(axes, results.items()):
        for arm, probs in arms_pull.items():
            ax.plot(x, probs, label=f'Arm {arm}')

        ax.set_title(f'Arm Selection Probability - {policy_name}')
        ax.set_xlabel('Horizon')
        ax.set_ylabel('P(arm selected)')
        ax.set_ylim(0, 1)
        set_integer_xticks(ax, horizon)
        ax.legend()

    plt.tight_layout()
    plt.show()

# def posterior_evolution(policy, env, rounds):
#     alpha = [1] * env.arms
#     beta_params = [1] * env.arms

#     for _ in range(rounds):
#         arm = policy.select_arm(env.arms)
#         reward, _ = env.pull(arm)
#         policy.update(arm, reward)

#         arm_index = arm - 1
#         alpha[arm_index] += reward
#         beta_params[arm_index] += (1 - reward)

#     x = np.linspace(0, 1, 100)
#     plt.figure(figsize=(10, 6))

#     for i in range(env.arms):
#         y = beta.pdf(x, alpha[i], beta_params[i])
#         plt.plot(x, y, label=f'Arm {i + 1}')

#     plt.title(f'Posterior Distributions after {rounds} Rounds - {policy.name}')
#     plt.xlabel('Reward Probability')
#     plt.ylabel('Density')
#     plt.legend()
#     plt.show()

def plot_regret_distribution(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    for ax, (policy_name, data) in zip(axes, results.items()):
        regrets_runs = np.array(data["regrets"])  # shape: (runs, horizon)

        # Final cumulative regret per run
        final_regrets = np.sum(regrets_runs, axis=1)

        ax.hist(final_regrets, bins=30, alpha=0.7)
        ax.set_title(f'Regret Distribution - {policy_name}')
        ax.set_xlabel('Final Cumulative Regret')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    horizon = 5000
    runs = 1000
    arms = 3
    arms_arr = [1, 2, 3]

    env = Environment(name="Environment 1", arms=3, mus=[0.53, 0.55, 0.53])
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

    plot_results(avg_results, horizon)
    plot_pulls(avg_results, horizon)
    plot_regret_distribution(results)

