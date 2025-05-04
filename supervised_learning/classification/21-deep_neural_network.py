#!/usr/bin/env python3
"""
Define una red neuronal profunda para clasificación binaria
"""

import numpy as np


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

    def forward_prop(self, X):
        """
        Calcula la propagación hacia adelante de la red neuronal

        Parámetros:
        - X: np.ndarray de forma (nx, m) que contiene los datos de entrada
            - nx: número de características (features)
            - m: número de ejemplos

        Retorna:
        - La salida de la red neuronal después de la última capa
        - El diccionario cache con todas las activaciones intermedias
        """
        self.cache["A0"] = X

        for indice in range(1, self.L + 1):
            Wl = self.weights[f"W{indice}"]
            bl = self.weights[f"b{indice}"]
            A_prev = self.cache[f"A{indice - 1}"]

            Zl = np.matmul(Wl, A_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))  # Función sigmoide

            self.cache[f"A{indice}"] = Al

        return Al, self.cache

    def cost(self, Y, A):
        """
        Calcula el costo de la red neuronal usando la función log loss

        Parámetros:
        - Y: np.ndarray de shape (1, m) con las etiquetas verdaderas
        - A: np.ndarray de shape (1, m) con las predicciones del modelo

        Retorna:
        - cost: el valor del costo
        """
        m = Y.shape[1]
        zz = 1.0000001 - A
        cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(zz))
        return cost

    def evaluate(self, X, Y):
        """
        Evalúa el rendimiento de la red neuronal

        Parámetros:
        - X: np.ndarray de shape (nx, m) con los datos de entrada
        - Y: np.ndarray de shape (1, m) con las etiquetas verdaderas

        Retorna:
        - predicción: np.ndarray de shape (1, m) con valores 0 o 1
        - costo: el costo del modelo con respecto a Y
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

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