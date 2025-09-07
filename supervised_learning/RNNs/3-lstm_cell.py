#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


def softmax(x):
    """asdasdasdas"""
    expx = np.exp(x - np.max(x, axis=1, keepdims=True))
    return expx / np.sum(expx, axis=1, keepdims=True)


class LSTMCell:
    """asdasdasdas"""
    def __init__(self, i, h, o):
        """
        qdsadasdas
        """
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))

        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))

        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))

        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))

        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        asdasdsadsadas
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        ft = 1 / (1 + np.exp(-(concat @ self.Wf + self.bf)))
        ut = 1 / (1 + np.exp(-(concat @ self.Wu + self.bu)))
        ctil = np.tanh(concat @ self.Wc + self.bc)

        cnext = ft * c_prev + ut * ctil
        ot = 1 / (1 + np.exp(-(concat @ self.Wo + self.bo)))
        hnext = ot * np.tanh(cnext)
        y = softmax(hnext @ self.Wy + self.by)

        return hnext, cnext, y
