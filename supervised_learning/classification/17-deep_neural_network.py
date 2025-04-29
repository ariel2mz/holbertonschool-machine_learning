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
