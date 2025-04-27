#!/usr/bin/env python3
"""Neuron module"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """ Adsdsadsadsa """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        for i in range(0, nodes):
            self.W1[i] = np.random.randn(1, nx)
            self.b1[i] = 0
            self.A1[i]= 0
        self.W2 = np.random.randn(1, nx)
        self.b2 = 0
        self.A2 = 0