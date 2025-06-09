#!/usr/bin/env python3
"""asdsadsa"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    sadsadsa sadsadas
    """
    init = K.initializers.HeNormal(seed=0)
    compressed_filters = int(nb_filters * compression)

    bn = K.layers.BatchNormalization(axis=-1)(X)
    relu = K.layers.Activation('relu')(bn)
    conv = K.layers.Conv2D(
        filters=compressed_filters, kernel_size=1,
        padding='same', kernel_initializer=init
    )(relu)
    pool = K.layers.AveragePooling2D(pool_size=2, strides=2)(conv)

    return pool, compressed_filters
