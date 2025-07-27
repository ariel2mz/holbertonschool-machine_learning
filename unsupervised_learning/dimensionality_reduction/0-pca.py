#!/usr/bin/env python3
"""sadsadsadasdsa"""
import numpy as np


def pca(X, var=0.95):
    """
    dasdsadsa
    """
    X_centered = X - np.mean(X, axis=0)
    covm = np.cov(X_centered, rowvar=False)
    eivals, eivecs = np.linalg.eigh(covm)
    sortedidx = np.argsort(eivals)[::-1]
    eivals = eivals[sortedidx]
    eivecs = eivecs[:, sortedidx]
    cuvar = np.cumsum(eivals) / np.sum(eivals)
    nd = np.argmax(cuvar >= var) + 1

    return eivecs[:, :nd]
