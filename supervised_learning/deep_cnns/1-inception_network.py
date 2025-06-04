#!/usr/bin/env python3
"""Builds the Inception network using Keras"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the GoogLeNet (Inception v1) architecture (2014).

    Returns:
        Keras model
    """
    input_layer = K.Input(shape=(224, 224, 3))

    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        activation='relu')(input_layer)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(X)
    X = K.layers.Conv2D(64, (1, 1), padding='same',
                        activation='relu')(X)
    X = K.layers.Conv2D(192, (3, 3), padding='same',
                        activation='relu')(X)
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(X)

    X = inception_block(X, [64, 96, 128, 16, 32, 32])
    X = inception_block(X, [128, 128, 192, 32, 96, 64])
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(X)

    X = inception_block(X, [192, 96, 208, 16, 48, 64])
    X = inception_block(X, [160, 112, 224, 24, 64, 64])
    X = inception_block(X, [128, 128, 256, 24, 64, 64])
    X = inception_block(X, [112, 144, 288, 32, 64, 64])
    X = inception_block(X, [256, 160, 320, 32, 128, 128])
    X = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                              padding='same')(X)

    X = inception_block(X, [256, 160, 320, 32, 128, 128])
    X = inception_block(X, [384, 192, 384, 48, 128, 128])

    X = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                  padding='valid')(X)
    X = K.layers.Dropout(0.4)(X)
    X = K.layers.Flatten()(X)
    out = K.layers.Dense(1000, activation='softmax')(X)

    model = K.Model(inputs=input_layer, outputs=out)

    return model
