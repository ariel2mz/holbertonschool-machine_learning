#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def pca(X, var=0.95):
    """
    sadadsada
    """
    X_centered = X - np.mean(X, axis=0)
    no, sv, rsv = np.linalg.svd(X_centered, full_matrices=False)

    expvar = (sv ** 2) / (X_centered.shape[0] - 1)
    totvar = np.sum(expvar)
    cumvar = np.cumsum(expvar) / totvar

    n = np.argmax(cumvar >= var) + 1

    return rsv[:n].T
