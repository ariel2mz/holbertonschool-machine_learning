#!/usr/bin/env python3
"""
sadasdasdsad
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Parameters:
    - confusion: np.ndarray of shape (classes, classes)

    Returns:
    - np.ndarray of shape (classes,) containing the specificity of each class
    """
    classes = confusion.shape[0]
    total = np.sum(confusion)
    lista = np.zeros(classes)

    for i in range(classes):
        corpos = confusion[i, i]
        falsopos = np.sum(confusion[:, i]) - corpos
        falsoneg = np.sum(confusion[i, :]) - corpos
        corneg = total - (corpos + falsopos + falsoneg)

        if (corneg + falsopos) > 0:
            lista[i] = corneg / (corneg + falsopos)
        else:
            lista[i] = 0

    return lista
