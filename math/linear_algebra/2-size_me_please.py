#!/usr/bin/env python3
def matrix_shape(matrix):
    """
    Determines the shape (dimensions) of a given matrix.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list: A list of integers representing the size of each dimension.
              - [rows] for a 1D list.
              - [rows, cols] for a 2D matrix.
              - [rows, cols, depth] for a 3D matrix.
              and more for a 4d
    """
    size = [len(matrix)]
    if len(matrix) > 0 and isinstance(matrix[0], list):
        size.append(len(matrix[0]))
        if len(matrix[0]) > 0 and isinstance(matrix[0][0], list):
            size.append(len(matrix[0][0]))
            if len(matrix[0][0]) > 0 and isinstance(matrix[0][0][0], list):
                size.append(len(matrix[0][0][0]))
    return size
