#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    asdfghgfds
    asdfgfds
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    vc = v / (1 - beta1 ** t)
    sc = s / (1 - beta2 ** t)
    var -= alpha * vc / (np.sqrt(sc) + epsilon)

    return var, v, s