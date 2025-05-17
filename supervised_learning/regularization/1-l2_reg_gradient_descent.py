#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Performs one pass of gradient descent on the neural network
    with L2 regularization.
    """
    np.set_printoptions(precision=8, suppress=False, floatmode='maxprec_equal')

    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        dW = (1 / m) * np.matmul(dz, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        # Update weights and biases
        weights['W' + str(l)] = W - alpha * dW
        weights['b' + str(l)] = b - alpha * db

        if l > 1:
            dz_prev = np.matmul(W.T, dz) * (1 - A_prev ** 2)
            dz = dz_prev
