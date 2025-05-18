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

    for layer in range(L, 0, -1):
        A_prev = cache[f"A{layer - 1}"]

        # Compute gradients
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights with learning rate
        weights[f"W{layer}"] -= alpha * dW
        weights[f"b{layer}"] -= alpha * db

        if layer > 1:  # If not the input layer
            # Backpropagate through dropout and tanh
            dA_prev = np.dot(weights[f"W{layer}"].T, dZ)
            dA_prev *= cache[f"D{layer - 1}"]   
            dA_prev /= keep_prob  # Scale to maintain expected value

            # Derivative of tanh activation: (1 - A^2)
            dZ = dA_prev * (1 - np.power(A_prev, 2))

    return weights  # Return updated weights
