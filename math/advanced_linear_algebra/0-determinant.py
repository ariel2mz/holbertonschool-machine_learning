#!/usr/bin/env python3
"""
sadsadsadsa
dsadsadsadsa
"""


def determinant(matrix):
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    n = len(matrix)
    # tiene q ser cuadrada para poder hallar determinante
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for col in range(n):
        # arma una matriz mas chica
        minor = [
            [matrix[i][j] for j in range(n) if j != col]
            for i in range(1, n)
        ]
        # cambia los signos porque es -1 elevado x (va alternando)
        signo = (-1) ** col
        det += signo * matrix[0][col] * determinant(minor)
    return det
