#!/usr/bin/env python3
"""asdsadas adss"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights in-place using gradient descent with inverted dropout.

    Args:
        Y (np.ndarray): One-hot labels of shape (classes, m).
        weights (dict): Contains 'W1'..'WL', 'b1'..'bL'.
        cache (dict): Contains 'A0'..'AL' and 'D1'..'DL-1'.
        alpha (float): Learning rate.
        keep_prob (float): Probability a node is kept during dropout.
        L (int): Number of layers.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db

        if i > 1:
            dA_prev = np.matmul(weights[f"W{i}"].T, dZ)
            dA_prev *= cache[f"D{i-1}"]
            dA_prev /= keep_prob
            dZ = dA_prev * (1 - A_prev**2)
