import numpy as np

class Environment:
    def __init__(self, name="Environment", arms=3, mus=[0.9, 0.8, 0.7], seed=None):
        self.reset(name, arms, mus, seed)

    def reset(self, name, arms, mus, seed=None):
        """Reset environment after each experiment"""
        self.name = name
        self.arms = arms
        if len(mus) != arms:
            raise ValueError("Length of mus must be equal to number of arms.")
        self.mus = mus
        self.mu_star = max(mus)
        self.seed = seed or np.random.randint(0, 10000)
        self.rng = np.random.default_rng(self.seed)

    def pull(self, arm):
        """Pull the selected arm and return reward"""
        if arm < 1 or arm > self.arms:
            raise ValueError("Arm index out of range.")
        arm -= 1  # Adjust for zero-based indexing
        reward = self.rng.binomial(1, self.mus[arm])
        regret = self.mu_star - self.mus[arm]
        return reward, regret