#!/usr/bin/env python3

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
"""
el proyecto 2 dla semana uwu
"""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    hago un modelo cmo ates pero ahora con keras
    q lo hace mucho mas facil
    """
    model = Sequential()

    model.add(Dense(layers[0], input_dim=nx, activation=activations[0],
                    kernel_regularizer=l2(lambtha)))
    model.add(Dropout(1 - keep_prob))

    for i in range(1, len(layers)):
        model.add(Dense(layers[i], activation=activations[i],
                        kernel_regularizer=l2(lambtha)))
        model.add(Dropout(1 - keep_prob))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
