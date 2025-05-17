#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent with L2 regularization.

    Parameters:
    Y -- one-hot numpy.ndarray of shape (classes, m) with correct labels
    weights -- dictionary of weights and biases
    cache -- dictionary of outputs of each layer
    alpha -- learning rate
    lambtha -- L2 regularization parameter
    L -- number of layers in the network
    """
    m = Y.shape[1]
    grads = {}

    dZ = cache['A' + str(L)] - Y

    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l-1)]

        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * weights['W' + str(l)]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        grads['dW' + str(l)] = dW
        grads['db' + str(l)] = db
        
        if l > 1:
            W = weights['W' + str(l)]
            dA_prev = np.dot(W.T, dZ)
            tanh_derivative = 1 - np.power(A_prev, 2)
            dZ = dA_prev * tanh_derivative

    for l in range(1, L+1):
        weights['W' + str(l)] -= alpha * grads['dW' + str(l)]
        weights['b' + str(l)] -= alpha * grads['db' + str(l)]