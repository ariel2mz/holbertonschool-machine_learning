#!/usr/bin/env python3
"""Algoritmo EM completo para GMM"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    ejecuta el algoritmo EM completo para ajustar un GMM a los datos
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0 or
            not isinstance(tol, float) or tol < 0 or
            not isinstance(verbose, bool)):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None
    g, lant = expectation(X, pi, m, S)
    if g is None or lant is None:
        return None, None, None, None, None

    for i in range(iterations):
        # paso M: actualizamos pi, m, S usando g
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # paso E: obtenemos nuevo g y nueva log likelihood
        g, lact = expectation(X, pi, m, S)
        if g is None or lact is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {lact:.5f}")
        if abs(lact - lant) <= tol:
            if verbose and i % 10 != 0:
                print(f"Log Likelihood after {i} iterations: {lact:.5f}")
            break

        lant = lact

    return pi, m, S, g, lact
