#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


class RNNCell:
    def __init__(self, i, h, o):
        """
        <zxsdfghjkl
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        sdfghjklñ
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        hnext = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        yraw = np.matmul(hnext, self.Wy) + self.by
        exp = np.exp(yraw - np.max(yraw, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return hnext, y
