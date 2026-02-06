#!/usr/bin/env python3
"""Bayesian Optimization Module"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialize Bayesian Optimization
        
        Args:
            f: Black-box function to optimize
            X_init: Initial sampled inputs (t, 1)
            Y_init: Outputs for X_init (t, 1)
            bounds: Tuple (min, max) of search space
            ac_samples: Number of acquisition samples
            l: Kernel length parameter
            sigma_f: Kernel output std deviation
            xsi: Exploration-exploitation factor
            minimize: True for minimization, False for maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.best = np.min(Y_init) if minimize else np.max(Y_init)

    def acquisition(self):
        """
        Calculate the next best sample location using Expected Improvement
        
        Returns:
            X_next: Next sample point to evaluate
            EI: Expected Improvement values across the search space
        """
        from scipy.stats import norm

        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            improvement = self.best - mu - self.xsi
        else:
            improvement = mu - self.best - self.xsi

        Z = improvement / sigma
        EI = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)

        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Run Bayesian optimization for specified number of iterations
        
        Args:
            iterations: Number of optimization steps
            
        Returns:
            X_opt: Optimal input found
            Y_opt: Optimal output found
        """
        for _ in range(iterations):

            X_next, _ = self.acquisition()

            Y_next = self.f(X_next)

            self.gp.update(X_next, Y_next)

            if self.minimize:
                self.best = min(self.best, Y_next)
            else:
                self.best = max(self.best, Y_next)

        if self.minimize:
            opt_idx = np.argmin(self.gp.Y)
        else:
            opt_idx = np.argmax(self.gp.Y)
        
        X_opt = self.gp.X[opt_idx]
        Y_opt = self.gp.Y[opt_idx]
        
        return X_opt, Y_opt
