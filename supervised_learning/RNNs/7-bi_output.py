#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import numpy as np


class BidirectionalCell:
    """
    sadasdasdsadasdsa
    """
    def __init__(self, i, h, o):
        """
        sadasdasdas
        """

        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        sadsadsdadsa
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Whf + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        asdasdsdaa
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(concat @ self.Whb + self.bhb)
        return h_prev

    def output(self, H):
        """
        sdadasdsadsa
        """
        t, m, _ = H.shape

        Y = []
        for time_step in range(t):

            hconcat = H[time_step]

            output = hconcat @ self.Wy + self.by

            expoutput = np.exp(output - np.max(output, axis=1, keepdims=True))
            softmaxoutput = expoutput / np.sum(expoutput, axis=1, keepdims=True)
            
            Y.append(softmaxoutput)

        Y = np.array(Y)
        return Y
