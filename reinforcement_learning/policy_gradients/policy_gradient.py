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

    act = np.random.choice(probs.shape[1], p=probs[0])

    ones = np.zeros_like(probs)
    ones[0, act] = 1

    grad = state.T.dot(onest - probs)
    
    return int(act), grad
