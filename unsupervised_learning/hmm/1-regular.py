#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def regular(P):
    """
    asdsadsadsa
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n = P.shape[0]
    if P.shape[1] != n:
        return None
    if not np.allclose(P.sum(axis=1), 1):
        return None

    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1
    
    try:
        sstate = np.linalg.lstsq(A, b, rcond=None)[0]
        return sstate.reshape(1, n)
    except Exception:
        return None
