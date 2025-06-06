#!/usr/bin/env python3
"""sadsadsads"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    sadfdsdadfgf sadsad
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    camA = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(s, s),
                           padding='valid', kernel_initializer=init)(A_prev)
    camA = K.layers.BatchNormalization(axis=3)(camA)
    camA = K.layers.Activation('relu')(camA)

    camA = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), strides=(1, 1),
                           padding='same', kernel_initializer=init)(camA)
    camA = K.layers.BatchNormalization(axis=3)(camA)
    camA = K.layers.Activation('relu')(camA)

    camA = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1),
                           padding='valid', kernel_initializer=init)(camA)
    camA = K.layers.BatchNormalization(axis=3)(camA)

    camB = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(s, s),
                           padding='valid', kernel_initializer=init)(A_prev)
    camB = K.layers.BatchNormalization(axis=3)(camB)

    final = K.layers.Add()([camA, camB])
    final = K.layers.Activation('relu')(final)

    return final
