#!/usr/bin/env python3
import numpy as np

class DeepNeuralNetwork:
    """Deep neural network class for binary classification"""

    def __init__(self, nx, layers):
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)            # Number of layers
        self.cache = {}                 # To store forward prop values
        self.weights = {}               # To store weights and biases

        l = 0
        while l < self.L:
            layer_input_size = nx if l == 0 else layers[l - 1]
            self.weights[f'W{l + 1}'] = (
                np.random.randn(layers[l], layer_input_size) *
                np.sqrt(2 / layer_input_size)
            )
            self.weights[f'b{l + 1}'] = np.zeros((layers[l], 1))
            l += 1