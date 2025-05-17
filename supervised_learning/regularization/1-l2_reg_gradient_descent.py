#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights of a NN using GD with L2

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
    lm = lambtha
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i-1)]
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lm / m) * weights['W' + str(i)]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        grads['dW' + str(i)] = dW
        grads['db' + str(i)] = db

        if i > 1:
            W = weights['W' + str(i)]
            dA_prev = np.dot(W.T, dZ)
            tanh_derivative = 1 - np.power(A_prev, 2)
            dZ = dA_prev * tanh_derivative

    for i in range(1, L+1):
        weights['W' + str(i)] -= alpha * grads['dW' + str(i)]
        weights['b' + str(i)] -= alpha * grads['db' + str(i)]
