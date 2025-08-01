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

        # vectoriza restando cada punto con cada centroide
        X_vectors = np.repeat(X[:, np.newaxis], k, axis=1)  # (n, k, d)
        C_vectors = np.tile(cents[np.newaxis, :], (n, 1, 1))  # (n, k, d)

        # calcula la distancia euclidea entre cada punto y cada centro
        distances = np.linalg.norm(X_vectors - C_vectors, axis=2)

        # asigna a cada dato cual es el cluster mas cercano que tiene
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

        cents = nuevocents

    return cents, clss
