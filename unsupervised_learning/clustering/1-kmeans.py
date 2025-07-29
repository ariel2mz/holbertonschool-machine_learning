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


# el checker decia que estaba mal se lo pase a la IA
# dice que los cluster mios estan en distinto orden,
# me dio esta funcion que los ordena
def sort_centroids_and_labels_no_loops(centroids, labels):
    # Step 1: Sort centroids by their coordinates (e.g., lexsort by all dimensions)
    sorted_indices = np.lexsort(centroids.T)
    sorted_centroids = centroids[sorted_indices]

    # Step 2: Create a mapping from old index to new index (inverse sort)
    inverse_map = np.zeros_like(sorted_indices)
    inverse_map[sorted_indices] = np.arange(len(sorted_indices))

    # Step 3: Remap labels using the inverse map
    new_labels = inverse_map[labels]

    return sorted_centroids, new_labels


def kmeans(X, k, iterations=1000):
    """
    sadsadadsads
    """
    # clasico check
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(k, int) or k <= 0 or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None
    
    minvals = np.min(X, axis=0)
    maxvals = np.max(X, axis=0)

    # inicializo el centroid con la funcion anterior
    # y si la funcion falla retorna none entonces coso pum
    cents = initialize(X, k)
    if cents is None:
        return None, None

    for i in range(iterations):

        # calcula la distancia de cada dato a cada cluster
        distances = np.linalg.norm(X[:, np.newaxis] - cents, axis=2)

        # asigna a cada dato cual es el cluster mas cercano que tiene
        clss = np.argmin(distances, axis=1)

        # guarda el clustering para comparar si cambio, para cortar
        # la iteracion antes de tiempo
        centsrespaldo = cents.copy()

        # recorre cada centro (cluster)
        for j in range(k):

            # guarda en points todos los puntos del centro actual j
            points = X[clss == j]

            if points.shape[0] > 0:

                # pone el centro en el medio de todos los puntos
                cents[j] = np.mean(points, axis=0)

            else:

                # reinicia el cluster a otro lado si no tiene puntos asignados
                cents[j] = np.random.uniform(low=minvals, high=maxvals)

        # si ningun centro cambio, paras la  iteracion porque no tiene sentido
        if np.allclose(centsrespaldo, cents):
            break

    # la IA me dio esta funcion para ordenar los centroids para el checker
    sorted_centroids, new_labels = sort_centroids_and_labels(cents, clss)
    return sorted_centroids, new_labels
