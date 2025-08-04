#!/usr/bin/env python3
"""asdsadsadsaa"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    asdsadasda
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(kmin, int) or kmin < 1 or
            (kmax is not None and (not isinstance(kmax, int)
                                   or kmax < kmin)) or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    likes = []
    bics = []
    totales = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, llike = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose)

        if pi is None or m is None or S is None or g is None or llike is None:
            return None, None, None, None

        p = k * d + k * d * (d + 1) / 2 + (k - 1)

        bic = p * np.log(n) - 2 * llike

        totales.append((pi, m, S))
        likes.append(llike)
        bics.append(bic)

    likes = np.array(likes)
    bics = np.array(bics)

    bid = np.argmin(bics)
    bk = kmin + bid
    bresult = totales[bid]

    return bk, bresult, likes, bics
