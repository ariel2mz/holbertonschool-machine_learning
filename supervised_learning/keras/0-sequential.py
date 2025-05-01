#!/usr/bin/env python3

import tensorflow.keras as K
"""
sasdsadsa
"""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    hace un modelo con keras
    """
    model = K.Sequential()

    model.add(K.layers.Dense(layers[0], input_dim=nx, activation=activations[0],
                             kernel_regularizer=K.regularizers.l2(lambtha),
                             input_shape=(nx,)))

    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(lambtha)))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
