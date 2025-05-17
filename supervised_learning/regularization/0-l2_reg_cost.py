#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    - cost: float, original cost of the network without regularization
    - lambtha: float, L2 regularization parameter (lambda)
    - weights: dict, contains weights and biases of the network)
    - L: int, number of layers
    - m: int, number of data points
    
    Returns:
    total: float, the cost including L2 regularization
    """
    l2 = 0
    for i in range(1, L + 1):
        W = weights["W" + str(i)]
        l2 += np.sum(np.square(W))

    total = cost + (lambtha / (2 * m)) * l2

    return total
