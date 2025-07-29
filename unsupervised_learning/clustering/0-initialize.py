#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def initialize(X, k):
    """
    resumen en la cuadernola
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    no, d = X.shape
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)

    try:
        cent = np.random.uniform(low=min, high=max, size=(k, d))
        return cent

    except Exception:
        return None
