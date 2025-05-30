#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Creates mini-batches from (X, Y) for mini-batch gradient descent."""
    X_shuffled, Y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
