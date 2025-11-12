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
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return prob
