#!/usr/bin/env python3
"""
sadasdasdsad
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout regularization
    using gradient descent:
    """
    grds = {}
    m = Y.shape[1]
    A_output = cache['A' + str(L)]
    dZ = A_output - Y 

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i-1)]
        grds['db' + str(i)] = np.sum(dZ, axis=1, keepdims=True) / m
        grds['dW' + str(i)] = np.dot(dZ, A_prev.T) / m

        if i != 1: 
            dA = np.matmul(weights['W' + str(i)].T, dZ)
            dA *= cache['D' + str(i-1)]
            dA /= keep_prob
            dZ = dA * (1 - np.power(A_prev, 2))

        weights['b' + str(i)] -= alpha * grds['db' + str(i)]
        weights['W' + str(i)] -= alpha * grds['dW' + str(i)]
