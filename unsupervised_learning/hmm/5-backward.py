#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    sadsadsadsadsa
    """
    Om = Observation
    Em = Emission
    Tm = Transition
    In = Initial
    if not isinstance(Om, np.ndarray) or Om.ndim != 1:
        return None, None
    if not isinstance(Em, np.ndarray) or Em.ndim != 2:
        return None, None
    if not isinstance(Tm, np.ndarray) or Tm.ndim != 2:
        return None, None
    if not isinstance(In, np.ndarray) or In.ndim != 2:
        return None, None

    T = Om.shape[0]
    N = Em.shape[0]

    if Tm.shape[0] != N or Tm.shape[1] != N:
        return None, None
    if In.shape[0] != N or In.shape[1] != 1:
        return None, None

    B = np.zeros((N, T))

    B[:, T-1] = 1.0

    for t in range(T-2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                Tm[i, :] *
                Em[:, Om[t+1]] *
                B[:, t+1]
            )

    P = np.sum(
        In[:, 0] *
        Em[:, Om[0]] *
        B[:, 0]
    )

    return P, B
