import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

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