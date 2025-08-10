#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    sDADSADSA
    """

    if (not isinstance(Observation, np.ndarray) or Observation.ndim != 1 or
            not isinstance(Emission, np.ndarray) or Emission.ndim != 2 or
            not isinstance(Transition, np.ndarray) or Transition.ndim != 2 or
            not isinstance(Initial, np.ndarray) or Initial.ndim != 2):
        return None, None

    Om = Observation
    Em = Emission
    Tm = Transition
    In = Initial
    T = Observation.shape[0]
    N, M = Emission.shape

    if Tm.shape != (N, N) or In.shape != (N, 1):
        return None, None
    if not np.allclose(Tm.sum(axis=1), 1) or not np.allclose(
            Em.sum(axis=1), 1):
        return None, None
    if not np.isclose(In.sum(), 1):
        return None, None

    F = np.zeros((N, T))
    for i in range(N):
        F[i, 0] = In[i, 0] * Em[i, Om[0]]

    for t in range(1, T):
        for j in range(N):
            suma = 0.0
            for i in range(N):
                suma += F[i, t-1] * Tm[i, j]
            F[j, t] = suma * Em[j, Om[t]]

    P = np.sum(F[:, -1])
    return P, F
