#!/usr/bin/env python3
"""
asdsaddas
"""
import numpy as np


def mean_cov(X):
    """
    sadsadas
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)

    """
    formula para obtener el cov
    """
    Xcent = X - mean
    cov = (Xcent.T @ Xcent) / (X.shape[0] - 1)

    return mean, cov
