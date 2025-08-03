#!/usr/bin/env python3
"""Expectation step del EM para GMM"""
import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Hace el paso de expectativa del algoritmo EM para un GMM
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(pi, np.ndarray) or len(pi.shape) != 1 or
            not isinstance(m, np.ndarray) or len(m.shape) != 2 or
            not isinstance(S, np.ndarray) or len(S.shape) != 3):
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if (m.shape != (k, d) or S.shape != (k, d, d)):
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    try:
        # Creamos una matriz g vacía para guardar las responsabilidades
        # (probabilidades posteriores) — shape (k, n)
        g = np.zeros((k, n))

        # Recorremos cada cluster
        for i in range(k):
            # Calculamos la probabilidad de cada punto según la gaussiana i
            P = pdf(X, m[i], S[i])
            if P is None:
                return None, None

            # Multiplicamos por la probabilidad previa del cluster i (pi[i])
            g[i] = pi[i] * P

        # Sumamos todas las probabilidades para cada punto (para normalizar)
        total = np.sum(g, axis=0)  # shape: (n,)

        # Si algún punto tiene prob 0 total, algo salió mal
        if np.any(total == 0):
            return None, None

        # Normalizamos: dividimos cada responsabilidad por el total del punto
        g /= total

        # Calculamos el log likelihood total
        like = np.sum(np.log(total))

        return g, like

    except Exception:
        return None, None
