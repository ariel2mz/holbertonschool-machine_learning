#!/usr/bin/env python3
"""fghkl"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


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
        sadsadsadas
        """

        Ks = self.kernel(self.X, X_s)
        Kss = self.kernel(X_s, X_s)
        Kinv = np.linalg.inv(self.K)

        mu = Ks.T.dot(Kinv).dot(self.Y).reshape(-1)

        covs = Kss - Ks.T.dot(Kinv).dot(Ks)

        return mu, np.diag(covs)

    def update(self, X_new, Y_new):
        """
        asdfghjk
        """
        self.X = np.vstack((self.X, X_new.reshape(-1, 1)))
        self.Y = np.vstack((self.Y, Y_new.reshape(-1, 1)))
        self.K = self.kernel(self.X, self.X)
