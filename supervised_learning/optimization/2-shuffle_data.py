#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way."""
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
