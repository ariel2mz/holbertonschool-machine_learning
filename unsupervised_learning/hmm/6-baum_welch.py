#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    dsadsadsadsa
    """
    Om = Observations
    Em = Emission
    Tm = Transition
    In = Initial
    if not isinstance(Om, np.ndarray) or Om.ndim != 1:
        return None, None
    if not isinstance(Tm, np.ndarray) or Tm.ndim != 2:
        return None, None
    if not isinstance(Em, np.ndarray) or Em.ndim != 2:
        return None, None
    if not isinstance(In, np.ndarray) or In.ndim != 2:
        return None, None

    M = Tm.shape[0]
    N = Em.shape[1]
    T = Om.shape[0]

    if Tm.shape != (M, M):
        return None, None
    if Em.shape != (M, N):
        return None, None
    if In.shape != (M, 1):
        return None, None

    for _ in range(iterations):
        alpha = np.zeros((M, T))
        c = np.zeros(T)

        alpha[:, 0] = In[:, 0] * Em[:, Om[0]]
        c[0] = 1.0 / np.sum(alpha[:, 0])
        alpha[:, 0] *= c[0]

        # Recursion
        for t in range(1, T):
            for j in range(M):
                alpha[j, t] = Em[j, Om[t]] * np.sum(
                    alpha[:, t-1] * Tm[:, j]
                )
            c[t] = 1.0 / np.sum(alpha[:, t])
            alpha[:, t] *= c[t]

        beta = np.zeros((M, T))
        beta[:, T-1] = 1.0 * c[T-1]

        for t in range(T-2, -1, -1):
            for i in range(M):
                beta[i, t] = np.sum(
                    Tm[i, :] * Em[:, Om[t+1]] * beta[:, t+1]
                )
            beta[:, t] *= c[t]

        # Compute gamma and xi
        gamma = np.zeros((M, T))
        xi = np.zeros((M, M, T-1))
        for t in range(T-1):
            abv = beta[:, t+1].reshape(-1, 1)
            denom = np.sum(
                alpha[:, t] * Tm * Em[:, Om[t+1]].reshape(-1, 1) * abv
            )
            for i in range(M):
                gamma[i, t] = np.sum(
                    alpha[i, t] * Tm[i, :] * Em[:, Om[t+1]] * beta[:, t+1]
                ) / denom
                for j in range(M):
                    xi[i, j, t] = (
                        alpha[i, t] * Tm[i, j] * Em[j, Om[t+1]] * beta[j, t+1]
                    ) / denom

        # Special case for gamma at T-1
        gamma[:, T-1] = alpha[:, T-1] * beta[:, T-1] / np.sum(
            alpha[:, T-1] * beta[:, T-1]
        )

        In = gamma[:, 0].reshape(-1, 1)

        for i in range(M):
            for j in range(M):
                Tm[i, j] = np.sum(xi[i, j, :]) / np.sum(gamma[i, :-1])

        for j in range(M):
            for k in range(N):
                mask = (Om == k)
                Em[j, k] = np.sum(gamma[j, mask]) / np.sum(gamma[j, :])
    
    return Tm, Em
