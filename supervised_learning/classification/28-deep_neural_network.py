import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """Red neuronal profunda para clasificación multiclase"""

    def __init__(self, nx, layers, activation='sig'):
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
        self.__activation = activation  # Store the activation function choice

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

    @property
    def activation(self):
        return self.__activation

    def _activate(self, Z):
        """Activa Z según la función de activación elegida"""
        if self.__activation == 'sig':
            return 1 / (1 + np.exp(-Z))
        elif self.__activation == 'tanh':
            return np.tanh(Z)

    def _activation_derivative(self, A):
        """Devuelve la derivada de la activación elegida"""
        if self.__activation == 'sig':
            return A * (1 - A)
        elif self.__activation == 'tanh':
            return 1 - A ** 2

    def forward_prop(self, X):
        """Propagación hacia adelante con activación seleccionada en capas ocultas"""
        self.cache["A0"] = X

        for i in range(1, self.L + 1):
            W = self.weights[f"W{i}"]
            b = self.weights[f"b{i}"]
            A_prev = self.cache[f"A{i-1}"]

            Z = np.matmul(W, A_prev) + b

            if i == self.L:
                # softmax en la última capa
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                A = self._activate(Z)

            self.cache[f"A{i}"] = A

        return A, self.cache

    def cost(self, Y, A):
        """Costo usando cross-entropy multiclase"""
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
        """Descenso del gradiente con activación elegida en capas ocultas"""
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
                dZ = dA * self._activation_derivative(A)

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(W.T, dZ)

            self.weights[f"W{i}"] -= alpha * dW
            self.weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
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
