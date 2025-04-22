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

    @property
    def W(self):
        """
        Get the weights vector of the neuron.

        Returns:
            numpy.ndarray: The weights vector.
        """
        return self.__W

    @property
    def b(self):
        """
        Get the bias of the neuron.

        Returns:
            float: The bias value.
        """
        return self.__b

    @property
    def A(self):
        """
        Get the activated output of the neuron.

        Returns:
            float: The activated output.
        """
        return self.__A

    def forward_prop(self, X):
        """
    Calculate the forward propagation of the neuron.

    Args:
        X (numpy.ndarray): Input data of shape (nx, m).

    Returns:
        numpy.ndarray: The activated output (prediction) for each example.
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculate the cost using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels of shape (1, m).
            A (numpy.ndarray): Activated outputs of shape (1, m).

        Returns:
            float: The cost.
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neuron’s predictions.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).

        Returns:
            tuple: (predictions, cost)
                - predictions is a numpy.ndarray with predicted labels (0 or 1).
                - cost is the logistic regression cost.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, cost
