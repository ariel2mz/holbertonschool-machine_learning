#!/usr/bin/env python3
"""
asdsadsadsadsadsa sadsada
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    sadsadsadsa
    sadsadsa
    asdsads
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    nuevoh = int((h_prev + 2 * ph - kh) / sh) + 1
    nuevow = int((w_prev + 2 * pw - kw) / sw) + 1

    padd = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    nuevo = np.zeros((m, nuevoh, nuevow, c_new))

    for k in range(c_new):
        for i in range(nuevoh):
            hstart = i * sh
            for j in range(nuevow):
                wstart = j * sw
                region = padd[:, hstart:hstart+kh, wstart:wstart+kw, :]
                nuevo[:, i, j, k] = np.sum(region * W[..., k], axis=(1, 2, 3))
                nuevo[:, i, j, k] = nuevo[:, i, j, k] + b[..., k]

    return activation(nuevo)
