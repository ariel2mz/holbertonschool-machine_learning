#!/usr/bin/env python3
"""
Define una red neuronal profunda para clasificación binaria
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """
    Clase que define una red neuronal profunda para clasificación binaria
    """

    def __init__(self, nx, layers):
        """
        Constructor de la clase

        Parámetros:
        - nx (int): Cantidad de características de entrada (features)
        - layers (list): Lista que representa el número de nodos

        Atributos:
        - L (int): Número de capas en la red neuronal
        - cache (dict): Almacena todos los valores intermedios
        - weights (dict): Almacena los pesos y sesgos (biases) de la red

        Excepciones:
        - TypeError: Si nx no es un entero
        - ValueError: Si nx es menor que 1
        - TypeError: Si layers no es una lista de enteros positivos
        """
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

            # He et al
            self.__weights[f"W{i + 1}"] = (
                np.random.randn(layers[i], prev) * np.sqrt(2 / prev)
            )
            self.__weights[f"b{i + 1}"] = np.zeros((layers[i], 1))
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

    def save(self, filename):
        """Guarda el objeto de la instancia en un archivo en formato pickle"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Carga un objeto DeepNeuralNetwork desde un archivo pickle"""
        if not os.path.isfile(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)

    def forward_prop(self, X):
        """
        Propagación hacia adelante con softmax para clasificación multiclase.
        """
        self.cache["A0"] = X

        for i in range(1, self.L + 1):
            W = self.weights[f"W{i}"]
            b = self.weights[f"b{i}"]
            A_prev = self.cache[f"A{i - 1}"]

            Z = np.matmul(W, A_prev) + b

            if i == self.L:
                # Softmax en la última capa
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                # Sigmoide en capas ocultas
                A = 1 / (1 + np.exp(-Z))

            self.cache[f"A{i}"] = A

        return A, self.cache

    def cost(self, Y, A):
        """
        Costo usando cross-entropy para clasificación multiclase.
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 0.0000001)) / m  # Evitar log(0)
        return cost

    def evaluate(self, X, Y):
        """
        Evalúa la red neuronal para clasificación multiclase.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.argmax(A, axis=0)
        labels = np.argmax(Y, axis=0)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Realiza un paso de descenso del gradiente para toda la red profunda

        Parámetros:
        - Y: numpy.ndarray con forma (1, m) que contiene las etiquetas
        - cache: diccionario que contiene todas las activaciones A{l}
        - alpha: tasa de aprendizaje

        Actualiza los pesos y bias en self.weights
        """
        m = Y.shape[1]
        L = self.L
        weights = self.weights.copy()
        dZ = 0

        for lar in reversed(range(1, L + 1)):
            A = cache["A" + str(lar)]
            A_prev = cache["A" + str(lar - 1)]
            W = weights["W" + str(lar)]

            if lar == L:
                dZ = A - Y
            else:
                dZ = dA * (A * (1 - A))

            dW = (1 / m) * np.matmul(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.matmul(W.T, dZ)

            self.weights["W" + str(lar)] -= alpha * dW
            self.weights["b" + str(lar)] -= alpha * db

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """
        Entrena e imprimir y graficar el costo.

        Parámetros:
        - X: np.ndarray (nx, m) con datos de entrada
        - Y: np.ndarray (1, m) con etiquetas correctas
        - iterations: número de iteraciones
        - alpha: tasa de aprendizaje
        - verbose: si es True, imprime el costo cada 'step' iteraciones
        - graph: si es True, grafica el costo al final
        - step: intervalo para imprimir/graficar

        Retorna:
        - Evaluación del modelo tras el entrenamiento
        """

        # Validaciones
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
            AL, cache = self.forward_prop(X)
            cost = self.cost(Y, AL)

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
