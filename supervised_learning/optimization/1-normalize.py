#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def normalize(X, m, s):
    """
    X: matriz
    m: media
    s: deviation

    Returns: matriz pasada por la formula
    """
    for i in range(0, len(X)):
        for j in range(0, len(X[0])):
            X[i][j] = (X[i][j] - m[j]) / s[j]
    
    return X