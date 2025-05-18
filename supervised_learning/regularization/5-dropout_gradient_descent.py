#!/usr/bin/env python3
"""asdsadas adss"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates `weights` (in‑place) using one step of gradient descent
    for a network that employed inverted Dropout during forward‑prop.

    Parameters
    ----------
    Y : ndarray (classes, m)
        One‑hot labels.
    weights : dict
        Keys 'W1'..'WL', 'b1'..'bL'.
    cache : dict
        Contains A0..AL and D1..D(L‑1) from forward pass.
    alpha : float
        Learning‑rate.
    keep_prob : float
        Probability a node was kept (same as in forward pass).
    L : int
        Number of layers.
    """
    m = Y.shape[1]

    dZ = cache[f"A{L}"] - Y

    for l in range(L, 0, -1):
        A_prev = cache[f"A{l-1}"]
        W_key, b_key = f"W{l}", f"b{l}"

        dW = (1 / m) * dZ @ A_prev.T
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights[W_key] -= alpha * dW
        weights[b_key] -= alpha * db

        if l == 1:
            break

        dA_prev = weights[W_key].T @ dZ
        D_prev = cache[f"D{l-1}"]
        dA_prev *= D_prev
        dA_prev /= keep_prob
        A_prev = cache[f"A{l-1}"]
        dZ = dA_prev * (1 - A_prev**2)
