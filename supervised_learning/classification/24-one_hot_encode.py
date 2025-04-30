#!/usr/bin/env python3
"""One-hot encoding function"""

import numpy as np

def one_hot_encode(Y, classes):
    """
    Convierte un vector de etiquetas numéricas en una matriz one-hot.

    Parámetros:
    - Y: np.ndarray de forma (m,), etiquetas numéricas
    - classes: número total de clases

    Retorna:
    - np.ndarray de forma (classes, m) con codificación one-hot, o None
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if classes <= 0 or np.any(Y >= classes) or np.any(Y < 0):
        return None

    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None
