#!/usr/bin/env python3
"""Bayesian Optimization implementation"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init,
                 Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        agshdjkjfsj
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0],
                               bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        asdasdasdsadsa
        """
        mu, sigma = self.gp.predict(self.X_s)
        sigma = sigma**0.5
        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        Z = np.zeros_like(mu)
        with np.errstate(divide='warn'):
            Z = np.where(sigma > 0, imp / sigma, 0)

        EI = np.where(
            sigma > 0,
            imp * norm.cdf(Z) + sigma * norm.pdf(Z),
            0
        )

        next = self.X_s[np.argmax(EI)]

        return next, EI
