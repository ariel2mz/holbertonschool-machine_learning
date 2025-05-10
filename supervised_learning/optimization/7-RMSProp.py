#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm

    Parameters:
    - alpha: learning rate (float)
    - beta2: RMSProp weight (float, between 0 and 1)
    - epsilon: small number to avoid division by zero (float)
    - var: numpy.ndarray, the variable to update
    - grad: numpy.ndarray, the gradient of var
    - s: numpy.ndarray, the previous second moment of var

    Returns:
    - The updated variable and the new second moment (s)
    """
    s = beta2 * s + (1 - beta2) * np.square(grad)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
