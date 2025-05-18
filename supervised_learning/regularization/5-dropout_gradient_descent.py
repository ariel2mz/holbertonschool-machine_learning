#!/usr/bin/env python3
"""Gradient descent with inverted dropout regularization"""
import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Perform one pass of gradient descent with dropout regularization.

    Parameters:
    - Y: np.ndarray one-hot labels, shape (classes, m)
    - weights: dict, weights and biases with keys 'W1'...'WL' and 'b1'...'bL'
    - cache: dict, cached activations 'A0'...'AL' and dropout masks 'D1'...'D(L-1)'
    - alpha: float, learning rate
    - keep_prob: float, probability of keeping a node
    - L: int, number of layers

    Updates weights in place.
    """
    m = Y.shape[1]
    dZ_current = cache[f"A{L}"] - Y  # Gradient at output layer

    for layer in range(L, 0, -1):
        A_prev = cache[f"A{layer -1}"]

        dW = (1 / m) * dZ_current @ A_prev.T
        db = (1 / m) * np.sum(dZ_current, axis=1, keepdims=True)

        # Update weights and biases
        weights[f"W{layer}"] -= alpha * dW
        weights[f"b{layer}"] -= alpha * db

        if layer == 1:
            break  # No layer before input layer to backpropagate to

        # Backpropagate through dropout and tanh activation
        dA_prev = weights[f"W{layer}"].T @ dZ_current
        dA_prev *= cache[f"D{layer -1}"]       # Apply dropout mask
        dA_prev /= keep_prob                    # Scale to compensate dropout

        # Derivative of tanh activation: (1 - A^2)
        dZ_current = dA_prev * (1 - np.power(A_prev, 2))
