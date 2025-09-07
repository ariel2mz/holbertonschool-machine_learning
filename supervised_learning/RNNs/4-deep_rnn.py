#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    asdadsa)
    """
    cells = rnn_cells
    t, m, i = X.shape
    l, _, h = h_0.shape
    o = cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    for step in range(t):
        xinput = X[step]
        for layer in range(l):
            hprev = H[step, layer]
            hnext, y = rnn_cells[layer].forward(hprev, xinput)
            H[step + 1, layer] = hnext
            xinput = hnext
        Y[step] = y

    return H, Y
