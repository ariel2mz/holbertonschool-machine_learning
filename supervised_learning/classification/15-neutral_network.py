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

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Forward propagation"""
        Z = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z))

        # en vez de multiplicar W con X se mutiplica con la A1 que saque
        # de la hidden layer

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

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
        cost = self.cost(Y, A[1])
        prediction = np.where(A[1] >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Performs one pass of gradient descent"""
        m = Y.shape[1]
        dW2 = (1 / m) * np.matmul(A2 - Y, A1.T)
        db2 = (1 / m) * np.sum(A2 - Y, axis=1, keepdims=True)
        dZ1 = np.matmul(self.W2.T, A2 - Y) * A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2

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