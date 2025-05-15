#!/usr/bin/env python3
""" asdasdsadasdas """
import numpy as np


def precision(confusion):
    """
    Calcula la precisión para cada clase en una matriz de confusión.

    Parámetro:
        confusion (np.ndarray): matriz de confusión (clases, clases)

    Retorna:
        np.ndarray: arreglo con la precisión por clase
    """
    corr = np.diag(confusion)
    todos = np.sum(confusion, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.true_divide(corr, todos)
        precision[np.isnan(precision)] = 0
    return precision
