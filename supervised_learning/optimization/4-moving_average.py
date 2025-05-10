#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average with bias correction.

    Parameters:
    - data: list of float or int values
    - beta: float, weight factor between 0 and 1

    Returns:
    - list of moving averages with bias correction
    """
    averages = []
    v = 0
    for i, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        bias_corrected = v / (1 - beta ** (i + 1))
        averages.append(bias_corrected)
    return averages
