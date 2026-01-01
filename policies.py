import numpy as np

class Policy: 
    def __init__(self, name, arms, rules, seed=None):
        self.reset(name, arms, rules, seed)

    def select_arm(self, arms):
        """Arm selection rules"""
        if not arms or arms == 0:
            raise ValueError("Arms cannot be 0.")
        return self.rules(arms)
    
    def update(self, arm, reward):
        """Update after arm pull"""
        arm_index = arm - 1  # Adjust for zero-based indexing
        self.total_counts[arm_index] += 1
        self.reward_counts[arm_index] += reward

    def reset(self, name, arms, rules, seed=None):
        """Reset after each experiment"""
        self.name = name
        self.rules = rules
        self.reward_counts = [0]*arms
        self.total_counts = [0]*arms
        self.seed = seed or np.random.randint(0, 10000)
        self.rng = np.random.default_rng(self.seed)

class GreedyPolicy(Policy):
    def __init__(self, arms):
        def rules(arms):
            mu_hat = (np.array(self.reward_counts) + 1) / (np.array(self.total_counts) + 2)
            return np.argmax(mu_hat) + 1  # Adjust for one-based indexing
        super().__init__("Greedy Policy", arms, rules)

class e_GreedyPolicy(Policy):
    def __init__(self, arms, epsilon=0.1):
        def rules(arms):
            if self.rng.random() <= epsilon:
                return self.rng.integers(1, arms + 1)  # Explore
            
            mu_hat = (np.array(self.reward_counts) + 1) / (np.array(self.total_counts) + 2)
            return np.argmax(mu_hat) + 1  # Exploit
        super().__init__("Epsilon-Greedy Policy", arms, rules)

class UCBPolicy(Policy):
    def __init__(self, arms):
        self.mu_hat = [0]*arms
        def rules(arms):
            untried = [i for i, n in enumerate(self.total_counts) if n == 0]
            if untried:
                return np.random.choice(untried) + 1  # Explore untried arms
            
            t = sum(self.total_counts)          # guaranteed t >= arms >= 1
            n = np.array(self.total_counts)     # guaranteed n_i >= 1
            bonuses = np.sqrt((2 * np.log(t)) / n)
            ucb_values = np.array(self.mu_hat) + bonuses
            return np.argmax(ucb_values) + 1
        super().__init__("UCB Policy", arms, rules)

    def update(self, arm, reward):
        super().update(arm, reward)
        self.mu_hat[arm - 1] += (reward - self.mu_hat[arm - 1]) / self.total_counts[arm - 1]

class ThompsonSamplingPolicy(Policy):
    def __init__(self, arms):
        self.alpha = [1]*arms
        self.beta = [1]*arms
        def rules(arms):
            sampled_values = [self.rng.beta(self.alpha[i], self.beta[i]) for i in range(arms)]    
            return np.argmax(sampled_values) + 1
        super().__init__("Thompson Sampling Policy", arms, rules)

    def update(self, arm, reward):
        super().update(arm, reward)
        if reward == 1:
            self.alpha[arm-1] += 1
        else:
            self.beta[arm-1] += 1