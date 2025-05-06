#!/usr/bin/env python3
import numpy as np

def normalization_constants(X):
    vcolum = 0
    pcolum = []
    stdcolum = []
    suma_cuadrados = 0

    ccolum = len(X[0])
    for j in range(0, ccolum):
        for i in range(0, len(X)):
            vcolum = vcolum + X[i][j]
        pcolum.append(float(vcolum / len(X)))
        vcolum = 0

    for j in range(0, ccolum):
        for i in range(0, len(X)):
            resta = X[i][j] - pcolum[j]
            suma_cuadrados += resta ** 2
        desviacion = (suma_cuadrados / len(X)) ** 0.5
        stdcolum.append(float(desviacion))
        suma_cuadrados = 0

    return pcolum, stdcolum