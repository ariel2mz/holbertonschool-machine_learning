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
    A_output = cache['A' + str(L)]
    dZ = A_output - Y  # Gradient for the output layer (softmax + cross-entropy)

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        
        # Compute gradients for weights and biases
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights using the learning rate
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:  # If not the input layer
            # Backpropagate through the layer
            dA = np.dot(weights['W' + str(i)].T, dZ)
            dA *= cache['D' + str(i - 1)]  # Apply dropout mask
            dA /= keep_prob  # Scale to maintain expected value
            dZ = dA * (1 - np.power(A_prev, 2))  # Derivative of tanh
