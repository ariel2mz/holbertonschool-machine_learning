#!/usr/bin/env python3
"""
asdsaddas
"""
import numpy as np


def correlation(C):
    """
    Calculate the correlation matrix from a covariance matrix.

    Parameters:
    C (numpy.ndarray): Covariance matrix of shape (d, d)

    Returns:
    numpy.ndarray: Correlation matrix of shape (d, d)

    Raises:
    TypeError: If C is not a numpy.ndarray
    ValueError: If C is not a 2D square matrix
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    variances = np.diag(C)

    if np.any(variances <= 0):
        raise ValueError("a")

    std_dev = np.sqrt(variances)
    std_dev_matrix = np.outer(std_dev, std_dev)
    correlation_matrix = C / std_dev_matrix

    return correlation_matrix
