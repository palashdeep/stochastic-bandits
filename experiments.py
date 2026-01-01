import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

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

def set_integer_xticks(ax, horizon, max_ticks=8):
    """Set integer x-ticks for a discrete horizon without clutter"""
    step = max(1, horizon // max_ticks)
    ticks = np.arange(0, horizon, step)

    ax.set_xticks(ticks)
    ax.set_xlim(0, horizon-1)

def plot_results(results, horizon=20):
    """Cumulative mean regret with/without std bands"""
    x = np.arange(horizon)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axes

    for policy_name, (regrets, std_regrets, _) in results.items():
        T = len(regrets)
        x = np.arange(T)

        # Cumulative mean regret
        cumulative_regrets = np.cumsum(regrets)
        cumulative_std = np.sqrt(np.cumsum(std_regrets ** 2))

        # cumulative regret
        ax1.plot(x, cumulative_regrets, label=policy_name)

        # cumulative regret with std bands
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
    """Plot of arm selection percentage"""
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

def plot_ts_posterior_evolution(alpha_history, beta_history, arm_index, timesteps):
    """Shows posterior evolution for TS for one run"""
    x = np.linspace(0, 1, 500)

    plt.figure(figsize=(8, 5))

    for t in timesteps:
        a = alpha_history[t][arm_index]
        b = beta_history[t][arm_index]
        y = beta.pdf(x, a, b)
        plt.plot(x, y, label=f"t = {t}")

    plt.xlabel("Mean reward")
    plt.ylabel("Density")
    plt.title(f"Posterior evolution (Arm {arm_index + 1})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_regret_distribution(results):
    """Plots histogram of final cumulative regret distribution"""
    labels = list(results.keys())
    data = [np.sum(np.array(results[label]["regrets"]), axis=1) for label in labels]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Final Cumulative regret")
    plt.title("Regret variability across runs")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Experiment 1: To compare different policies

    horizon = 5000
    runs = 1000
    arms = 3
    arms_arr = [1, 2, 3]

    env = Environment(name="Environment 1", arms=arms, mus=[0.53, 0.55, 0.53])
    policies = [
        GreedyPolicy(env.arms),
        e_GreedyPolicy(env.arms, epsilon=0.1),
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

    # Experiment 2: for TS posterior evolution

    horizon = 1000
    
    env = Environment(name="Environment 2", arms=3, mus=[0.8, 0.9, 0.7])
    policy = ThompsonSamplingPolicy(env.arms)

    arm_to_plot = 2
    timesteps = [10, 50, 200, 1000]

    alpha_hist, beta_hist = run_ts_experiment(env, policy, horizon)

    plot_ts_posterior_evolution(alpha_hist, beta_hist, arm_to_plot-1, timesteps)
