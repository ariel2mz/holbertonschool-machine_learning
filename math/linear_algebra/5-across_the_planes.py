#!/usr/bin/env python3
"""

ASDSADSADASDSA

"""


def add_matrices2D(mat1, mat2):
    """
    Determines the shape (dimensions) of a given matrix.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list:
    """
    height = len(mat1)
    width = len(mat1[0])
    if height != len(mat2):
        return None
    if width != len(mat2[0]):
        return None
    nueva = [[0] * width for _ in range(height)]
    for i in range(width):
        for j in range(height):
            nueva[i][j] = mat1[i][j] + mat2[i][j]
    return nueva
