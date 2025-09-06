#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    sadsadsadasdsa
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0

    hprev = h_0
    for step in range(t):
        hnext, y = rnn_cell.forward(hprev, X[step])
        H[step + 1] = hnext
        Y[step] = y
        hprev = hnext

    return H, Y
