#!/usr/bin/env python3
"""
Neuron implementation module.

This module defines a Neuron class used for binary classification,
including initialization of weights, bias, and activated output.
"""

import numpy as np


class Neuron:
    """
    Represents a single neuron performing binary classification.

    Attributes:
        W (numpy.ndarray): The weights vector for the neuron,
                           initialized using a random normal distribution.
        b (float): The bias for the neuron, initialized to 0.
        A (float): The activated output (prediction) of the neuron.
    """

    def __init__(self, nx):
        """
        Initialize a Neuron instance.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0
