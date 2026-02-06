#!/usr/bin/env python3
"""bayes_opt"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a 1D Gaussian process
    without noise. Uses Expected Improvement for acquisition.
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        f: black-box function to optimize
        X_init: numpy.ndarray of shape (t, 1), initial sample inputs
        Y_init: numpy.ndarray of shape (t, 1), initial sample outputs
        bounds: tuple (min, max) of search space
        ac_samples: number of candidate points for acquisition
        l, sigma_f: Gaussian Process hyperparameters
        xsi: exploration-exploitation parameter
        minimize: True for minimization, False for maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def _norm_pdf(self, x):
        """Standard normal probability density function"""
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

    def _norm_cdf(self, x):
        """Standard normal cumulative density function using erf"""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))

    def acquisition(self):
        """
        Computes the Expected Improvement (EI) for each candidate point
        and selects the next point to sample.
        Returns:
            X_next: numpy.ndarray of shape (1,)
            EI: numpy.ndarray of shape (ac_samples,)
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma.reshape(-1, 1)

        # Determine current best based on minimize/maximize
        if self.minimize:
            best = np.min(self.gp.Y)
            improvement = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            improvement = mu - best - self.xsi

        # Avoid division by zero
        EI = np.zeros_like(improvement)
        mask = sigma > 0
        Z = np.zeros_like(improvement)
        Z[mask] = improvement[mask] / sigma[mask]
        EI[mask] = improvement[mask] * self._norm_cdf(Z[mask]) + sigma[mask] * self._norm_pdf(Z[mask])

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI.ravel()

    def optimize(self, iterations=100):
        """
        Runs Bayesian optimization for a maximum number of iterations.
        Stops early if a proposed point has already been sampled.
        Returns:
            X_opt: numpy.ndarray of shape (1,) optimal input
            Y_opt: numpy.ndarray of shape (1,) optimal output
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Stop if X_next was already sampled
            if np.any(np.isclose(self.gp.X, X_next)):
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        # Find optimal sampled point
        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx].reshape(1,)
        Y_opt = self.gp.Y[idx].reshape(1,)
        return X_opt, Y_opt
