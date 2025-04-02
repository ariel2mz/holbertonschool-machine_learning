#!/usr/bin/env python3
"""

ASDSADSADASDSA

"""


def add_arrays(arr1, arr2):
    """
    Determines the shape (dimensions) of a given matrix.

    Args:
        matrix (list): A nested list representing the matrix.

    Returns:
        list:
    """
    leng = len(arr1)
    if leng != len(arr2):
        return None
    result = [0] * leng
    for i in range(leng):
        result[i] = arr1[i] + arr2[i]
    return result
