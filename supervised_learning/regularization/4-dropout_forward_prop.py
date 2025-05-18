#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Parameters:
    X (numpy.ndarray): Input data of shape (nx, m)
    weights (dict): Dictionary containing weights and biases of the neural network
    L (int): Number of layers in the network
    keep_prob (float): Probability that a node will be kept

    Returns:
    dict: A dictionary containing the outputs of each layer and the dropout mask used on each layer
    """
    cache = {}
    A = X
    m = X.shape[1]

    for l in range(1, L + 1):
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        Z = np.dot(W, A) + b
        if l < L:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(l)] = D
        else:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        cache['A' + str(l)] = A

    return cache
