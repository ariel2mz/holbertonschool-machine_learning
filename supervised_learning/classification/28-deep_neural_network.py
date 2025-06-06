#!/usr/bin/env python3
"""
Deep Neural Network for multiclass classification
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """Defines a deep neural network for multiclass classification."""

    def __init__(self, nx, layers, activation='sig'):
        """
        Constructor for the deep neural network.

        Parameters:
        - nx: number of input features
        - layers: list of nodes in each layer
        - activation: activation function ('sig' or 'tanh')
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        for i in range(self.L):
            layer_size = layers[i]
            if not isinstance(layer_size, int) or layer_size < 1:
                raise TypeError("layers must be a list of positive integers")

            prev_size = nx if i == 0 else layers[i - 1]
            self.__weights[f"W{i + 1}"] = np.random.randn(
                layer_size, prev_size) * np.sqrt(2 / prev_size)
            self.__weights[f"b{i + 1}"] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Returns number of layers."""
        return self.__L

    @property
    def cache(self):
        """Returns intermediate values (cache)."""
        return self.__cache

    @property
    def weights(self):
        """Returns weights and biases."""
        return self.__weights

    @property
    def activation(self):
        """Returns activation function name."""
        return self.__activation

    @staticmethod
    def activation_function(Z, activation='sig'):
        """
        Applies activation function to input Z.

        Parameters:
        - Z: linear input
        - activation: type of activation ('sig', 'tanh', 'softmax')
        """
        if activation == 'sig':
            return 1 / (1 + np.exp(-Z))
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'softmax':
            e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return e_Z / np.sum(e_Z, axis=0, keepdims=True)

    def forward_prop(self, X):
        """
        Performs forward propagation.

        Parameters:
        - X: input data of shape (nx, m)
        """
        self.__cache = {"A0": X}
        for i in range(1, self.L + 1):
            W = self.weights[f"W{i}"]
            b = self.weights[f"b{i}"]
            A_prev = self.cache[f"A{i - 1}"]
            Z = np.dot(W, A_prev) + b

            if i == self.L:
                A = self.activation_function(Z, 'softmax')
            else:
                A = self.activation_function(Z, self.activation)

            self.__cache[f"A{i}"] = A
        return A, self.__cache

    def cost(self, Y, A):
        """
        Computes cost using cross-entropy loss.

        Parameters:
        - Y: true labels (one-hot)
        - A: predicted output
        """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m

    def evaluate(self, X, Y):
        """
        Evaluates predictions and cost.

        Parameters:
        - X: input data
        - Y: true labels
        """
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        labels = np.argmax(Y, axis=0)
        cost = self.cost(Y, A)

        one_hot = np.zeros_like(A)
        one_hot[predictions, np.arange(A.shape[1])] = 1

        return one_hot, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Updates weights using gradient descent.

        Parameters:
        - Y: true labels
        - cache: cached values from forward prop
        - alpha: learning rate
        """
        m = Y.shape[1]
        L = self.L
        weights = self.weights
        A_L = cache[f"A{L}"]
        dZ = A_L - Y

        for i in reversed(range(1, L + 1)):
            A_prev = cache[f"A{i - 1}"]
            W = weights[f"W{i}"]

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                dA_prev = np.dot(W.T, dZ)
                if self.activation == 'sig':
                    dZ = dA_prev * A_prev * (1 - A_prev)
                elif self.activation == 'tanh':
                    dZ = dA_prev * (1 - A_prev ** 2)

            self.__weights[f"W{i}"] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.

        Parameters:
        - X: input data
        - Y: true labels
        - iterations: number of iterations
        - alpha: learning rate
        - verbose: prints cost every 'step'
        - graph: plots cost over iterations
        - step: step interval for verbose/graph
        """
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be a positive float")
        if verbose or graph:
            if not isinstance(step, int) or step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0:
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    steps.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance to a file.

        Parameters:
        - filename: name of the file to save to
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a saved instance.

        Parameters:
        - filename: file to load from
        """
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
