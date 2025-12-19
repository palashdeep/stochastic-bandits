# Stochastic Multi-Armed Bandits: Regret Minimization and Bayesian Exploration

This project studies the stochastic multi-armed bandit problem with unknown reward distributions. I compare Greedy, ε-Greedy, Upper Confidence Bound(UCB) and Thompson Sampling(TS) policies, focusing on how different treatments of uncertainty affect exploration, regret and learning dynamics. Rather than treating this as an optimization problem, the emphasis is on understanding why policies behave differently under noise and limited information.

## Problem

We consider a stochastic K-armed bandit with independent arms. Each arm i produces a Bernoulli reward with unknown mean μ<sub>i</sub>. At each time step, the agent selects an arm, observes a reward, and updates its policy. Performace is measured as expected cummulative regret relative to the optimal arm.

## Why Regret?

Regret measures opportunity cost of learning while acting. Unlike accuracy or reward variance, regret captures both early mistakes and long-run adaptations. A policy with low regret balances exploration and exploitation efficiently over time instead of maximizing for short-term reward.

## Policies

1. Greedy - Selects the arm with highest posterior mean. This serves as a baseline and illustrates failure mode of premature exploitation.
2. ε-Greedy - Introduces random exploration to greedy at a fixed rate. While simple, it explores indiscriminately and continues sampling clearly suboptimal arms.
3. Upper Confidence Bound (UCB) - Selects actions optimistically using upper confidence bounds derived from concentration inequalities. Exploration is forced until uncertainty is ruled out.
4. Thompson Sampling (TS) - Samples from the posterior distribution over arm means and selects the action that is optimal under the sampled hypothesis. Exploration emerges naturally in proportion to posterior uncertainty.

## Experimental Design

Experiments use a 3-armed Bernoulli bandit with carefully chosen mean rewards to expose the exploration-exploitation tradeoff. Each experiment runs for T=5000 steps and results are averaged over 1000 independent runs. All policies are evaluated on the same environments. Regret is computed using the true arm means ranther than the realized rewards to isolate learning behavior from noise.

## Results and Diagnostics

### Cummulative Regret

We plot mean cummulative regret over time to compare learning efficiency across policies.

### Action Selection Behavior

We analyze fraction of times each arm is selected to understand how policies allocate exploration

### Posterior Evolution (Thompson Sampling)

We visualize the evolution of posterior distributions for selected arms to illustrate how uncertainty collapses with data

### Regret Variability

We examine the distribution of final regret across runs to assess robustness

## Observations

- Greedy failure modes
- ε-Greedy inefficiency
- UCB vs TS tradeoffs
- Practical implications

## Limitation \& Extensions

This study focuses on independent stochastic bandits. Extensions could exploit structure across arms (e.g. linear or contextual bandits) or examine senstivity to prior misspecifications and heavy-tailed noise

## Relation to Decision-Making under Uncertainty

Bandit problems formalize sequential decision-making under uncertainty, where actions both generate reward and reveal information. This perspective closely mirrors real-world settings such as trading, where decisions must be made before uncertainty is resolved.




