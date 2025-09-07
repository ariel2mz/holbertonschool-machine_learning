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
