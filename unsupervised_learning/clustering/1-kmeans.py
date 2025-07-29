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

        diff = X[:, np.newaxis, :] - cents
        distances = np.sum(diff**2, axis=2)

        # asigna a cada dato cual es el cluster mas cercano que tiene
        clss = np.argmin(distances, axis=1)

        # guarda el clustering para comparar si cambio, para cortar
        # la iteracion antes de tiempo
        nuevocents = np.zeros_like(cents)

        # recorre cada centro (cluster)
        for j in range(k):

            # guarda en points todos los puntos del centro actual j
            points = X[clss == j]

            if points.shape[0] > 0:
                # pone el centro en el medio de todos los puntos
                nuevocents[j] = np.mean(points, axis=0)

            else:

                # reinicia el cluster a otro lado si no tiene puntos asignados
                nuevocents[j] = initialize(X, 1)[0]

        # si ningun centro cambio, paras la  iteracion porque no tiene sentido
        if np.all(cents == nuevocents):
            break

        diff = X[:, np.newaxis, :] - cents
        distances = np.sum(diff**2, axis=2)
        clss = np.argmin(distances, axis=1)

    return cents, clss
