#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def pca(X, ndim):
    """
    Transforms data into ndim dimensions using PCA
    
    Args:
        X: Input data matrix of shape (n, d)
        ndim: Target number of dimensions (must be <= d)
    
    Returns:
        Transformed data matrix of shape (n, ndim)
    """

    X_centered = X - np.mean(X, axis=0)
    Lval, Sval, Rval = np.linalg.svd(X_centered, full_matrices=False)
    W = Rval[:ndim].T
    T = np.dot(X_centered, W)

    return T
