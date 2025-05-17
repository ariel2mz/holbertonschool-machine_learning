#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    sadasdasdas
    sadasdasdsa

    asdsadsa
    sadas
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]

        dw = (1 / m) * np.matmul(dz, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        weights['W' + str(l)] -= alpha * dw
        weights['b' + str(l)] -= alpha * db

        if l > 1:
            dz = np.matmul(W.T, dz) * (1 - A_prev ** 2)
