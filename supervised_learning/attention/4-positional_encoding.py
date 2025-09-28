#!/usr/bin/env python3
"""
asdsads
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    puse las notas en la cuadernola al final :V
    """

    posenc = np.zeros((max_seq_len, dm))

    pos = np.arange(max_seq_len).reshape(-1, 1)

    dim = np.arange(dm)

    angrates = 1 / np.power(10000, (2 * (dim // 2)) / np.float32(dm))

    angrads = pos * angrates
    posenc[:, 0::2] = np.sin(angrads[:, 0::2])
    posenc[:, 1::2] = np.cos(angrads[:, 1::2])

    return posenc
