#!/usr/bin/env python3
"""
Define una red neuronal profunda para clasificación multiclase
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """
    Clase que define una red neuronal profunda para clasificación multiclase
    """

    def __init__(self, nx, layers):
        """
        Constructor

        nx: número de características de entrada
        layers: lista con el número de nodos por capa
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx
        for i, nodes in enumerate(layers, 1):
            self.__weights[f"W{i}"] = np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            self.__weights[f"b{i}"] = np.zeros((nodes, 1))
            prev = nodes

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
        """
        Propagación hacia adelante con softmax en la capa final
        """
        self.__cache["A0"] = X

        for i in range(1, self.L + 1):
            W = self.weights[f"W{i}"]
            b = self.weights[f"b{i}"]
            A_prev = self.cache[f"A{i - 1}"]
            Z = np.matmul(W, A_prev) + b

            if i == self.L:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                A = 1 / (1 + np.exp(-Z))  # sigmoide

            self.__cache[f"A{i}"] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Csadsa
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evalúa el rendimiento de la red neuronal
        """
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Descenso de gradiente para actualizar pesos y sesgos
        """
        m = Y.shape[1]
        L = self.L
        weights = self.weights.copy()

        for i in reversed(range(1, L + 1)):
            A = cache[f"A{i}"]
            A_prev = cache[f"A{i - 1}"]
            W = weights[f"W{i}"]

            if i == L:
                dZ = A - Y
            else:
                dZ = dA * A * (1 - A)

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA = np.matmul(W.T, dZ)

            self.__weights[f"W{i}"] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Entrena la red neuronal

        - verbose: imprime el costo
        - graph: muestra la gráfica del costo
        - step: cada cuántas iteraciones imprimir
        """
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be a positive float")
        if verbose or graph:
            if not isinstance(step, int) or step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")
            if graph and (i % step == 0 or i == iterations):
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
        """Guarda el modelo en un archivo pickle"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Carga el modelo desde un archivo pickle"""
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
