#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def absorbing(P):
    """
    asdsadsadasdsa
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n = P.shape[0]
    if P.shape[1] != n:
        return False
    if not np.allclose(P.sum(axis=1), 1):
        return False

    AS = []
    for i in range(n):
        if np.isclose(P[i, i], 1.0) and np.allclose(P[i], np.eye(n)[i]):
            AS.append(i)

    if not AS:
        return False

    reach = np.zeros((n, n), dtype=bool)
    reach[P > 0] = True

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if reach[i, j]:
                    continue
                if reach[i, k] and reach[k, j]:
                    reach[i, j] = True

    for i in range(n):
        can_reach_absorbing = False
        for a in AS:
            if reach[i, a]:
                reachAbs = True
                break
        if not reachAbs:
            return False

    return True
