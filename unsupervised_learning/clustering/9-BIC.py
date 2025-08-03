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
        (kmax is not None and (not isinstance(kmax, int) or kmax < kmin)) or
        not isinstance(iterations, int) or iterations <= 0 or
        not isinstance(tol, float) or tol < 0 or
        not isinstance(verbose, bool)):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n

    likelihoods = []
    bics = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_likelihood = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose)

        if pi is None or m is None or S is None or g is None or log_likelihood is None:
            return None, None, None, None

        p = k * d + k * d * (d + 1) / 2 + (k - 1)

        bic = p * np.log(n) - 2 * log_likelihood

        results.append((pi, m, S))
        likelihoods.append(log_likelihood)
        bics.append(bic)

    likelihoods = np.array(likelihoods)
    bics = np.array(bics)

    best_idx = np.argmin(bics)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, likelihoods, bics
