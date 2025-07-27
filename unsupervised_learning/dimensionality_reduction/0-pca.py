#!/usr/bin/env python3
"""sadsadsadasdsa"""
import numpy as np


def pca(X, var=0.95):
    """
    sadsadas
    """

    covm = np.cov(X, rowvar=False)
    eivals, eivecs = np.linalg.eigh(covm)

    sortedidx = np.argsort(eivals)[::-1]
    eivals = eivals[sortedidx]
    eivecs = eivecs[:, sortedidx]

    cuvar = np.cumsum(eivals) / np.sum(eivals)

    nd = np.searchsorted(cuvar, var)
    W = eivecs[:, :nd]

    return W
