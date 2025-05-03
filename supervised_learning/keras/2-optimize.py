#!/usr/bin/env python3

import tensorflow.keras as K
"""
This module builds a neural network using the Functional API in Keras.
"""


def optimize_model(network, alpha, beta1, beta2):
    """
    Builds a neural network model using the Keras Functional API.

    Parameters:
    - network: asdasdsa dsasa
    - alpha: asdsadsa asdas
    - beta1: asdsad sadsa
    - beta2: asdsadsad asddsa
    """
    adam = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optimizer=adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
