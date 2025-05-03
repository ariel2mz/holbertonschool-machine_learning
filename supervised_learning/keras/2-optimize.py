#!/usr/bin/env python3
"""
This module builds a neural network using the Functional API in Keras.
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Builds a neural network model using the Keras Functional API.

    Parameters:
    - network (keras.Model):sadsadsa sdadsa.
    - alpha (float): asdsadsadsa sdadsad.
    - beta1 (float): sadsadsad sadsadsa.
    - beta2 (float): sadasd sadsadsa dsa.

    Returns:
    - None: nada.
    """
    adam = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optimizer=adam,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
