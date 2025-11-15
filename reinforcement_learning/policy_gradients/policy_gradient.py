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

    # ∇logπ(a|s) = φ(s) * (1(a=a') - π(a'|s))
    prob = policy(state, weight)

    action = np.random.choice(len(prob[0]), p=prob[0])

    grad = np.zeros_like(weight)

    grad[:, action] = state


    prob = policy(state, weight)
    action = np.random.choice(len(prob), p=prob)

    for a in range(weight.shape[1]):
        grad[:, a] -= prob[a] * state

    return act, grad
