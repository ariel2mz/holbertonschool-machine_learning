#!/usr/bin/env python3
"""sadsadsadasdsa"""
import numpy as np


def pca(X, var=0.95):
    """
    dsadasdsa
    """

    X_centered = X - np.mean(X, axis=0)
    covm = np.cov(X_centered, rowvar=False)
    eivals, eivecs = np.linalg.eigh(covm)
    sortedidx = np.argsort(eivals)[::-1]
    eivals = eivals[sortedidx]
    eivecs = eivecs[:, sortedidx]
    # This step helps match the expected output signs
    for i in range(eivecs.shape[1]):
        if eivecs[0, i] < 0:
            eivecs[:, i] *= -1
    cuvar = np.cumsum(eivals) / np.sum(eivals)
    nd = np.argmax(cuvar >= var) + 1
    W = eivecs[:, :nd]

    return W