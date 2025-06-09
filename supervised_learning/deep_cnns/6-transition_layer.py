#!/usr/bin/env python3
"""asdsadsa"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    sadsadsa sadsadas
    """
    ini = K.initializers.HeNormal(seed=0)
    cpfilters = int(nb_filters * compression)

    bn = K.layers.BatchNormalization(axis=-1)(X)
    relu = K.layers.Activation('relu')(bn)
    conv = K.layers.Conv2D(
        filters=cpfilters, kernel_size=1,
        padding='same', kernel_initializer=ini
    )(relu)
    pooling = K.layers.AveragePooling2D(pool_size=2, strides=2)(conv)

    return pooling, cpfilters
