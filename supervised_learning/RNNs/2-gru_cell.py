#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def softmax(x):
    """dasdsadas"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class GRUCell:
    """
    sadsadsadsad
    """
    def __init__(self, i, h, o):
        """
        dasfdsnakdjsl
        """

        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        asdaaddsadasdsa
        """
        m = x_t.shape[0]
        concat = np.concatenate((h_prev, x_t), axis=1)
        z_t = 1 / (1 + np.exp(-(concat @ self.Wz + self.bz)))
        r_t = 1 / (1 + np.exp(-(concat @ self.Wr + self.br)))
        concat_reset = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.tanh(concat_reset @ self.Wh + self.bh)
        h_next = (1 - z_t) * h_prev + z_t * h_tilde
        y = softmax(h_next @ self.Wy + self.by)

        return h_next, y
