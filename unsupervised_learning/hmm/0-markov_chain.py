#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    asdfghjkjgfd
    """
    if (not isinstance(P, np.ndarray) or not isinstance(s, np.ndarray) or
            P.ndim != 2 or s.ndim != 2):
        return None
    n = P.shape[0]
    if P.shape[1] != n or s.shape != (1, n):
        return None
    if not isinstance(t, int) or t < 0:
        return None
    if not (np.allclose(P.sum(axis=1), 1) and np.allclose(s.sum(), 1)):
        return None

    try:
        P_t = np.linalg.matrix_power(P, t)
        return np.dot(s, P_t)
    except Exception:
        return None
