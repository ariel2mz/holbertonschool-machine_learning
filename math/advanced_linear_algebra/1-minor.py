#!/usr/bin/env python3
"""
sadsadsadsa
dsadsadsadsa
"""


def determinant(matrix):
    """
    sadfgffdsadfg
    ASSa
    """
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    x = len(matrix)
    # tiene q ser cuadrada para poder hallar determinante
    if not all(len(row) == x for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if x == 1:
        return matrix[0][0]
    if x == 2:
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

    det = 0
    for col in range(x):
        # arma una matriz mas chica
        minor = [
            [matrix[i][j] for j in range(x) if j != col]
            for i in range(1, x)
        ]
        # cambia los signos porque es -1 elevado x (va alternando)
        signo = (-1) ** col
        det += signo * matrix[0][col] * determinant(minor)
    return det


def minor(matrix):
    """
    sadaasdsadsa
    sdasdsadsa
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    if matrix == [[]]:
        return [[1]]

    minorm = []
    for i in range(n):
        row = []
        for j in range(n):
            subm = [
                [matrix[x][y] for y in range(n) if y != j]
                for x in range(n) if x != i
            ]
            row.append(determinant(subm))
        minorm.append(row)
    # un numero menor es 
    # determinante de la submatriz que se obtiene al 
    # eliminar la fila y la columna del elemento en cuestión

    # repitiendo este proceso por cada numero de la matriz
    # se puede formar una nueva matriz minorm
    return minorm
