#!/usr/bin/env python3
"""
This module builds a neural network using the Functional API in Keras.
"""

def one_hot(labels, classes=None):
    """
    sadsadsadsa
    """
    leng = max(labels)
    matrix = []

    for i in range(0, len(labels)):
        row = []
        for x in range(0, leng):
            if labels[i] == x:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)
