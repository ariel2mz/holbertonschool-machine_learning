#!/usr/bin/env python3
"""
sadsadsad
"""
import numpy as np


def variance(X, C):
    """
    asdsads
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if X.ndim != 2 or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    if X.size == 0 or C.size == 0:
        return None

    distances = np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2, axis=2)

    mindis = np.min(distances, axis=1)

    total = np.sum(mindis)
    
    return total
