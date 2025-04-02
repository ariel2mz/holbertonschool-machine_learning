#!/usr/bin/env python3
"""

ASDSADSADASDSA

"""


def matrix_transpose(matrix):
    """
    Determines the shape (dimensions) of a given matrix.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list:
    """
    rows = len(matrix)
    cols = len(matrix[0])

    nueva = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            nueva[j][i] = matrix[i][j]

    return nueva
