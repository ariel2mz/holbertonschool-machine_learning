#!/usr/bin/env python3
"""One-hot decoding function"""

import numpy as np

def one_hot_decode(one_hot):
    """
    Convierte una matriz one-hot en un vector de etiquetas numéricas.

    Parámetros:
    - one_hot: np.ndarray de forma (classes, m), codificación one-hot

    Retorna:
    - np.ndarray de forma (m,) con etiquetas numéricas, o None en caso de error
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
