#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes the unactivated output of a neural network using batch normalization.

    Parameters:
    - Z (numpy.ndarray): shape (m, n), input data to normalize
    - gamma (numpy.ndarray): shape (1, n), scale parameter
    - beta (numpy.ndarray): shape (1, n), offset parameter
    - epsilon (float): small constant to avoid division by zero

    Returns:
    - Z_norm (numpy.ndarray): normalized and scaled output
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    Z_n = (Z - mean) / np.sqrt(var + epsilon)
    Z_t = gamma * Z_n+ beta

    return Z_t
  
