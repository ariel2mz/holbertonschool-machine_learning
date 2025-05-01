#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """Red neuronal profunda para clasificación multiclase"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx
        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights[f"W{i+1}"] = np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            self.__weights[f"b{i+1}"] = np.zeros((layers[i], 1))
            prev = layers[i]

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Propagación hacia adelante (softmax en la última capa)"""
        self.cache["A0"] = X

        for i in range(1, self.L + 1):
            W = self.weights[f"W{i}"]
            b = self.weights[f"b{i}"]
            A_prev = self.cache[f"A{i-1}"]

            Z = np.matmul(W, A_prev) + b

            if i == self.L:
                # softmax
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-Z))  # sigmoid

            self.cache[f"A{i}"] = A

        return A, self.cache

    def cost(self, Y, A):
        """Costo usando softmax cross-entropy"""
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost

    def evaluate(self, X, Y):
        """Evalúa el modelo"""
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        Y_labels = np.argmax(Y, axis=0)
        cost = self.cost(Y, A)
        return prediction, cost


    def gradient_descent(self, Y, cache, alpha=0.05):
        """Descenso del gradiente para clasificación multiclase"""
        m = Y.shape[1]
        weights = self.weights.copy()
        L = self.L

        for i in reversed(range(1, L + 1)):
            A = cache[f"A{i}"]
            A_prev = cache[f"A{i-1}"]
            W = weights[f"W{i}"]

            if i == L:
                dZ = A - Y
            else:
                dZ = dA * (A * (1 - A))  # sigmoid derivative

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(W.T, dZ)

            self.weights[f"W{i}"] -= alpha * dW
            self.weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Entrena la red neuronal"""

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
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if (graph and i % step == 0) or i == iterations:
                costs.append(cost)
                steps.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Guarda la red en archivo .pkl"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Carga una red desde archivo .pkl"""
        if not os.path.isfile(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
