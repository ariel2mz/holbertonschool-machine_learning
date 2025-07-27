#!/usr/bin/env python3
"""
asdsadsadsadsadsa sadsada
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    asdsadsa
    sadsadsa
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    A_prev_pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))

    dApp = np.pad(
        dA_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for x in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    h_s = i * sh
                    h_e = h_s + kh
                    w_s = j * sw
                    w_e = w_s + kw

                    A_slice = A_prev_pad[x, h_s:h_e, w_s:w_e, :]

                    abrev = W[:, :, :, k] * dZ[x, i, j, k]
                    dApp[x, h_s:h_e, w_s:w_e, :] += abrev
                    dW[:, :, :, k] += A_slice * dZ[x, i, j, k]

    if padding == 'same':
        dA_prev = dApp[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dApp

    return dA_prev, dW, db
