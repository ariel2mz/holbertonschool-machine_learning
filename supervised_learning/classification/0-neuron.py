#!/usr/bin/env python3
import numpy as np
"""
Neuron implementation module.

adasdsadsadsadsa asddsadasdsa saddsadsa
asdasdsadsadsad sadsadsada asdasdsa

Classes:
    Neuron: neurona
"""


class Neuron:
    """
    Represents an internal node in the decision tree.

    Attributes:
        neuron (self): the neuron
        nx (nx): numeroooo.
        W (W): random entre 1 y nx.
        b (b): dasdasdasdasda
        a (A): asdasdsadsa dsadsadsa
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
