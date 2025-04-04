#!/usr/bin/env python3
"""

ASDSADSADASDSA

"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Determines the shape (dimensions) of a given matrix.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list:
    """
    nuevo = []
    if (axis == 0):
        if len(mat1[0]) != len(mat2[0]):
            return None
        for fila in mat1:
            nuevo.append(fila[:])
        for fila2 in mat2:
            nuevo.append(fila2[:])
        return nuevo

    if (axis == 1):

        if len(mat1) != len(mat2):
            return None

        for fila1, fila2 in zip(mat1, mat2):
            nueva_fila = fila1[:] + fila2[:]
            nuevo.append(nueva_fila)
        return nuevo
