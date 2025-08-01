#!/usr/bin/env python3
"""dtyukl"""
import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    sadasdsadsa
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(kmin, int) or kmin <= 0 or
        (kmax is not None and (not isinstance(kmax, int) or kmax <= 0)) or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n = X.shape[0]
    if kmax is None:
        kmax = n
    if kmin >= kmax:
        return None, None

    resultado = []
    dvars = []

    for k in range(kmin, kmax + 1):
        cents, clss = kmeans(X, k, iterations)
        if cents is None or clss is None:
            return None, None
        resultado.append((cents, clss))
        var = variance(X, cents)
        dvars.append(var)

   # Normalize variances based on the smallest k
    minvar = dvars[0]
    dvdiff = []
    for v in dvars:
        diff = minvar - v
        dvdiff.append(diff)
        dvars = dvdiff

    return resultado, dvars
