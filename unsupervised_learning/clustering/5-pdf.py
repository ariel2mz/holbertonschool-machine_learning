#!/usr/bin/env python3
"""PDF of multivariate Gaussian"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the PDF of a multivariate Gaussian distribution
    """
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(m, np.ndarray) or len(m.shape) != 1 or
                not isinstance(S, np.ndarray) or len(S.shape) != 2):
        return None

    noseusa, d = X.shape
    if m.shape[0] != d or S.shape != (d, d):
        return None

    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)
    except Exception:
        return None

    if det <= 0:
        return None

    normc = 1 / np.sqrt(((2 * np.pi) ** d) * det)
    diff = X - m


    exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
    P = normc * np.exp(exponent)
    """
    The Mahalanobis distance is used to measure how far each point is from the
    mean. The result is scaled by a normalization factor based on the
    determinant of S.
    """
    P = np.maximum(P, 1e-300)

    return P
