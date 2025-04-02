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
    width = len(mat1)
    height = len(mat1[0])
    if width != len(mat2):
        return None
    if height != len(mat2[0]):
        return None
    nueva = [[0] * height for _ in range(width)]
    for i in range(width):
        for j in range(height):
            nueva[i][j] = mat1[i][j] + mat2[i][j]
    return nueva
