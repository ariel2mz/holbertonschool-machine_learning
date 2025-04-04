#!/usr/bin/env python3
"""

ASDSADSADASDSA

"""


def mat_mul(mat1, mat2):
    """
    Determines the shape (dimensions) of a given matrix.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list:
    """
    if len(mat1[0]) != len(mat2):
        return None
    ancho = len(mat2[0])
    alto = len(mat1)
    nuevo = [[0] * ancho for _ in range(alto)]
    for x in range(len(mat2)):
        for i in range(0,len(mat1)):
            numero = 0
            for j in range(len(mat1[0])):
                numero = numero + (mat1[i][j] * mat2[j][x])
            nuevo[i][x] = numero

    return nuevo
