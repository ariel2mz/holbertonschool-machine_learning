#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent with momentum

    Parameters:
    - alpha: learning rate (float)
    - beta1: momentum weight (float, between 0 and 1)
    - var: numpy.ndarray, the variable to update
    - grad: numpy.ndarray, the gradient of var
    - v: numpy.ndarray, the previous momentum

    Returns:
    - The updated variable and the new momentum
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
