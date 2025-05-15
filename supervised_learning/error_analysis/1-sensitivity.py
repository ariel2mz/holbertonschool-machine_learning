#!/usr/bin/env python3
"""Calculates the sensitivity for each class in a confusion matrix"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates sensitivity (recall) for each class in a confusion matrix.

    Args:
        confusion (np.ndarray): shape (classes, classes)

    Returns:
        np.ndarray: shape (classes,) with sensitivity for each class
    """
    corr = np.diag(confusion)
    todos = np.sum(confusion, axis=1)

    # esto es por si divide entre 0 porque no hay samples
    with np.errstate(divide='ignore', invalid='ignore'):
        sensitivity = np.true_divide(corr, todos)
        sensitivity[np.isnan(sensitivity)] = 0

    return sensitivity
