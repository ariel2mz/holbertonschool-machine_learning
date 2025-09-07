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
        sadsadsa
        """
        t, m, _ = H.shape
        Y = softmax(H @ self.Wy + self.by)
        return Y

def softmax(x):
    """softmax"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
