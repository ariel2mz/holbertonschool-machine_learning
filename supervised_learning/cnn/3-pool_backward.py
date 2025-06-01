#!/usr/bin/env python3
"""
asdsadsadsadsadsa sadsada
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    sadasdasdas
    sdadsada
    """
    m, h_new, w_new, c = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dAp = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    h_s = h * sh
                    h_e = h_s + kh
                    w_s = w * sw
                    w_e = w_s + kw

                    a_prev_slice = A_prev[i, h_s:h_e, w_s:w_e, ch]

                    if mode == 'max':
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dAp[i, h_s:h_e, w_s:w_e, ch] += mask * dA[i, h, w, ch]

                    elif mode == 'avg':
                        da = dA[i, h, w, ch] / (kh * kw)
                        dAp[i, h_s:h_e, w_s:w_e, ch] += np.ones((kh, kw)) * da

    return dAp
