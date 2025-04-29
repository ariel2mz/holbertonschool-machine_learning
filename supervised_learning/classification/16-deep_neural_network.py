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
        - layers (list): Lista que representa el número de nodos en cada capa de la red

        Atributos:
        - L (int): Número de capas en la red neuronal
        - cache (dict): Almacena todos los valores intermedios del modelo durante la propagación hacia adelante
        - weights (dict): Almacena los pesos y sesgos (biases) de la red

        Excepciones:
        - TypeError: Si nx no es un entero
        - ValueError: Si nx es menor que 1
        - TypeError: Si layers no es una lista de enteros positivos
        """
        if not isinstance(nx, int):
            raise TypeError("nx debe ser un entero")
        if nx < 1:
            raise ValueError("nx debe ser un entero positivo")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers debe ser una lista de enteros positivos")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev_nodes = nx

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers debe ser una lista de enteros positivos")

            #He et al
            self.weights[f"W{i + 1}"] = (
                np.random.randn(layers[i], prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            self.weights[f"b{i + 1}"] = np.zeros((layers[i], 1))
            prev_nodes = layers[i]
