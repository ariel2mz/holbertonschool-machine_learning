#!/usr/bin/env python3
"""sadasdasdasdas"""
import sklearn.mixture


def gmm(X, k):
    """
    adsdsadasdas
    """
    model = sklearn.mixture.GaussianMixture(
        n_components=k, covariance_type='full').fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
