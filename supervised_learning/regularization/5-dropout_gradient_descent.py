#!/usr/bin/env python3
"""Gradient descent with inverted dropout regularization"""
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
    m = Y.shape[1]  # Number of examples
    dZ = cache[f"A{L}"] - Y  # Output layer gradient (softmax + cross-entropy)

    for l in range(L, 0, -1):
        A_prev = cache[f"A{l-1}"]
        
        # Compute gradients
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights with learning rate
        weights[f"W{l}"] -= alpha * dW
        weights[f"b{l}"] -= alpha * db

        if l == 1:  # No layer before input
            break

        # Backpropagate through dropout and tanh
        dA_prev = np.dot(weights[f"W{l}"].T, dZ)
        dA_prev = dA_prev * cache[f"D{l-1}"]  # Apply dropout mask
        dA_prev = dA_prev / keep_prob  # Inverted dropout scaling
        dZ = dA_prev * (1 - np.power(A_prev, 2))  # tanh derivative
