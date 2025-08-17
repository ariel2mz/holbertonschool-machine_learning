#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


class GaussianProcess:
    """
    asdsadsadsads
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        sadasdasdsadsa
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        asdasdasdsa
        """
        abv = np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        dist = np.sum(X1**2, 1).reshape(-1, 1) + abv
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * dist)

    def predict(self, X_s):
        """
        asdasdsadasd
        """

        Ks = self.kernel(X_s, self.X)
        Kss = self.kernel(X_s, X_s)
        mu = Ks @ self.K_inv @ self.Y
        mu = mu.reshape(-1)

        sigsquared = np.diag(Kss - Ks @ self.K_inv @ Ks.T)
        sigma = np.sqrt(sigsquared)

        return mu, sigma
