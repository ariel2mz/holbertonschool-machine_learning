#!/usr/bin/env python3
"""
asdsaddas
"""
import numpy as np
fact = np.math.factorial
ve = "x must be an integer that is greater than or equal to 0"


def likelihood(x, n, P):
    """
    ghjklkjhgfdfghjkl
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(ve)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    bicoef = fact(n) / (fact(x) * fact(n - x))
    lihood = bicoef * (P ** x) * ((1 - P) ** (n - x))

    return lihood
