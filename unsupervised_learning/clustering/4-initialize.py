#!/usr/bin/env python3
"""asdsadsadasdas"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    asdasdasdasdsadsa
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0):
        return None, None, None

    noseusa, d = X.shape

    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # priors
    pi = np.full(shape=(k,), fill_value=1 / k)

    # all identity matrices
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
