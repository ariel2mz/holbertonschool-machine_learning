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
        mu, var = self.gp.predict(self.X_s)
        sigma = np.sqrt(var)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        with np.errstate(divide='ignore'):
            Z = np.zeros_like(mu)
            mask = sigma > 0
            Z[mask] = imp[mask] / sigma[mask]

        EI = np.zeros_like(mu)
        abv = sigma[mask] * norm.pdf(Z[mask])
        EI[mask] = imp[mask] * norm.cdf(Z[mask]) + abv

        EI = np.maximum(EI, 0)

        Xnext = self.X_s[np.argmax(EI)].reshape(1,)

        return Xnext, EI