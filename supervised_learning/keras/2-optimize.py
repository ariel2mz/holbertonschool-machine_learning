#!/usr/bin/env python3

import tensorflow.keras as K
"""
This module builds a neural network using the Functional API in Keras.
"""


def optimize_model(network, alpha, beta1, beta2):
    adam = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
