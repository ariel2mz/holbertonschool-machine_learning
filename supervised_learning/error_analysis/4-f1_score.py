#!/usr/bin/env python3
"""
sadasdasdsad
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Args:
        confusion (np.ndarray): shape (classes, classes)

    Returns:
        np.ndarray: F1 score per class
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)

    # por si es dividir entre cero
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 * (prec * sens) / (prec + sens)
        f1[np.isnan(f1)] = 0

    return f1
