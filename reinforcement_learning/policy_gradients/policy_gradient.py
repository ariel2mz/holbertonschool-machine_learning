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

def policy_gradient(state, weight):
    """
    safsafsafsa
    """

    state = state.reshape(1, -1)

    probs = policy(state, weight)

    act = np.random.choice(probs.shape[1])

    ones = np.zero(prob.shape[1])
    ones[act] = 1

    grad = state.T @ (ones - prob)
    
    return int(act), grad
