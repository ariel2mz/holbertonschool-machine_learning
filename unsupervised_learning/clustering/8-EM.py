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

    # inicializamos los parámetros
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # paso E inicial
    g, lprev = expectation(X, pi, m, S)
    if g is None or lprev is None:
        return None, None, None, None, None

    if verbose:
        print(f"Log Likelihood after 0 iterations: {lprev:.5f}")

    for i in range(1, iterations + 1):
        # paso M
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # paso E
        g, lcurr = expectation(X, pi, m, S)
        if g is None or lcurr is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == iterations):
            print(f"Log Likelihood after {i} iterations: {lcurr:.5f}")

        if abs(lcurr - lprev) <= tol:
            if verbose and i % 10 != 0:
                print(f"Log Likelihood after {i} iterations: {lcurr:.5f}")
            break

        lprev = lcurr

    return pi, m, S, g, lcurr
