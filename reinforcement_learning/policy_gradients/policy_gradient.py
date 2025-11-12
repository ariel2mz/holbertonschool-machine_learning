#!/usr/bin/env python3
"""
asgsagsagsaa
"""
import numpy as np


def policy(matrix, weight):
    """
    askaslkfasklfsa
    """
    z = matrix.dot(weight)
    expz = np.exp(z - np.max(z, axis=1, keepdims=True))
    prob = expz / np.sum(expz, axis=1, keepdims=True)

    return prob
