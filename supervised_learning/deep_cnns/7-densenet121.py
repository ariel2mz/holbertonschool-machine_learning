#!/usr/bin/env python3
"""asdsadsa sadsae"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    asdsadsadsa dsadsadsa
    """
    init = K.initializers.HeNormal(seed=0)
    input = K.Input(shape=(224, 224, 3))

    bn = K.layers.BatchNormalization(axis=-1)(input)
    relu = K.layers.Activation('relu')(bn)
    conv = K.layers.Conv2D(
        64, (7, 7), strides=2, padding='same',
        kernel_initializer=init
    )(relu)
    pool = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                 padding='same')(conv)

    nb_filters = 64

    X, nb_filters = dense_block(pool, nb_filters, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    bnfinal = K.layers.BatchNormalization(axis=-1)(X)
    relufinal = K.layers.Activation('relu')(bnfinal)
    avg_pool = K.layers.AveragePooling2D(pool_size=7,
                                         strides=1,
                                         padding='valid')(relufinal)

    out = K.layers.Dense(
        1000, activation='softmax',
        kernel_initializer=init
    )(avg_pool)

    model = K.Model(inputs=input, outputs=out)
    return model
