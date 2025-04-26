#!/usr/bin/env python3
"""Neuron module"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Neuron for binary classification"""

    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """Forward propagation"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Cost function (log loss)"""
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Performs one pass of gradient descent"""
        m = Y.shape[1]
        dW = (1 / m) * np.dot(A - Y, X.T)
        db = (1 / m) * np.sum(A - Y)
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.

        Returns:
            tuple: (predictions, cost) after training.
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        i = 0
        cost = []
        numiteration = []
        for _ in range(iterations):
            A = self.forward_prop(X)
            cost.append(self.cost(Y, A))
            numiteration.append(i)
            if verbose and (i % step == 0 or i == 0 or i == iterations - 1):
                print("Cost after", numiteration[i], "iterations:", cost[i])
            i = i + 1
            self.gradient_descent(X, Y, A, alpha)

        x = numiteration
        y = cost

        plt.plot(x, y)
        plt.xlabel("iteration")
        plt.ylabel("cost")
        plt.title("Training Cost")
        plt.xlim(0, numiteration[-1])
        plt.ylim(0, cost[0])
        plt.xticks(np.arange(0, max(numiteration[-1])+1, 500))
        plt.yticks(np.arange(0, max(cost[0])+1, 1))
        plt.show()

        return self.evaluate(X, Y)