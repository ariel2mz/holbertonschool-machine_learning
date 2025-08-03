#!/usr/bin/env python3
"""maximization step del EM para GMM"""
import numpy as np


def maximization(X, g):
    """
    sadsadsadsad
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(g, np.ndarray) or len(g.shape) != 2):
        return None, None, None
    n, d = X.shape
    k, n2 = g.shape
    if n != n2:
        return None, None, None

    # sumamos todas las responsabilidades por cluster
    nk = np.sum(g, axis=1) 

    # si algún cluster no tiene responsabilidad
    if np.any(nk == 0):
        return None, None, None

    # calculamos los nuevos centroides como promedio
    m = (g @ X) / nk[:, np.newaxis]

    # inicializamos como ceros
    S = np.zeros((k, d, d))

    # recorremos cada cluster para calcular su matriz covariance
    for i in range(k):
        # restamos a cada punto su nuevo centro
        diff = X - m[i]
        # multiplicamo cada resta con la responsabilidad del cluster
        weighted = g[i][:, np.newaxis] * diff
        # calculamos la covariance del cluster
        S[i] = (weighted.T @ diff) / nk[i]

    pi = nk / n

    return pi, m, S
