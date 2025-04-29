#!/usr/bin/env python3
"""
Neuron implementation module.

This module defines a Neuron class used for binary classification,
including initialization of weights, bias, and activated output.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Represents a single neuron performing binary classification.

    Attributes:
        W (numpy.ndarray): The weights vector for the neuron,
                           initialized using a random normal distribution.
        b (float): The bias for the neuron, initialized to 0.
        A (float): The activated output (prediction) of the neuron.
    """

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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            layer_input_size = nx if l == 1 else layers[l - 2]
            self.weights[f'W{l}'] = (
                np.random.randn(layers[l - 1], layer_input_size) *
                np.sqrt(2 / layer_input_size)
            )
            self.weights[f'b{l}'] = np.zeros((layers[l - 1], 1))
