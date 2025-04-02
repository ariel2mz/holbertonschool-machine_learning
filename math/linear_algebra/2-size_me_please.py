#!/usr/bin/env python3
# adasdasdasdasd
def matrix_shape(matrix):
    # sadsadsadasdsa
    size = [len(matrix)]
    if len(matrix) > 0 and isinstance(matrix[0], list):
        size.append(len(matrix[0]))
        if len(matrix[0]) > 0 and isinstance(matrix[0][0], list):
            size.append(len(matrix[0][0]))
            if len(matrix[0][0]) > 0 and isinstance(matrix[0][0][0], list):
                size.append(len(matrix[0][0][0]))
    return size
