#!/usr/bin/env python3
"""sadasdsadsa"""
import numpy as np


def initialize(X, k):
    """
    resumen en la cuadernola
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    no, d = X.shape
    minvals = np.min(X, axis=0)
    maxvals = np.max(X, axis=0)

    try:
        cent = np.random.uniform(low=minvals, high=maxvals, size=(k, d))
        return cent
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """
    sadsadadsads
    """
    # clasico check
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape
    minvals = np.min(X, axis=0)
    maxvals = np.max(X, axis=0)

    # inicializo el centroid con la funcion anterior
    # y si la funcion falla retorna none entonces coso pum
    cents = initialize(X, k)
    if cents is None:
        return None, None

    for i in range(iterations):

        Xv = np.repeat(X[:, np.newaxis], k, axis=1)
        Xv = np.reshape(Xv, (X.shape[0], k, X.shape[1]))
        Cv = np.tile(cents[np.newaxis, :], (X.shape[0], 1, 1))
        Cv = np.reshape(Cv, (X.shape[0], k, X.shape[1]))
        distances = np.linalg.norm(Xv - Cv, axis=2)
        clss = np.argmin(distances, axis=1)

        # guarda el clustering para comparar si cambio, para cortar
        # la iteracion antes de tiempo
        nuevocents = np.copy(cents)

        for j in range(k):
            # puntos del centro j
            points = X[clss == j]
            if points.shape[0] > 0:
                nuevocents[j] = np.mean(points, axis=0)
            else:
                # si esta vacio, reinicializa ese cluster
                nuevocents[j] = np.random.uniform(low=minvals, high=maxvals, size=(d,))

        # si ningun centro cambio, paras la iteracion
        if np.all(cents == nuevocents):
            return cents, clss

    Cv = np.tile(cents, (X.shape[0], 1))
    Cv = Cv.reshape(X.shape[0], k, X.shape[1])
    distance = np.linalg.norm(Xv - cents, axis=2)
    clss = np.argmin(distance ** 2, axis=1)

    return cents, clss
