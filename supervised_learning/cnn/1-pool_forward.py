#!/usr/bin/env python3
"""
asdsadsadsadsadsa sadsada
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    hgsdfdaafga
    sadafgndfdfsds
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    nuevoh = (h_prev - kh) // sh + 1
    nuevow = (w_prev - kw) // sw + 1

    nuevo = np.zeros((m, nuevoh, nuevow, c_prev))

    for i in range(nuevoh):
        for j in range(nuevow):
            for c in range(c_prev):
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                region = A_prev[:, h_start:h_end, w_start:w_end, c]

                if mode == 'max':
                    nuevo[:, i, j, c] = np.max(region, axis=(1, 2))
                else:
                    nuevo[:, i, j, c] = np.mean(region, axis=(1, 2))

    return nuevo
