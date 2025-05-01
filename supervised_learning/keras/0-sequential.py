#!/usr/bin/env python3
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    nx: number of input features
    layers: list containing the number of nodes in each layer
    activations: list containing activation functions for each layer
    lambtha: L2 regularization parameter
    keep_prob: probability that a node will be kept for dropout

    Returns: Keras model
    """
    model = K.Sequential()

    model.add(K.layers.Dense(layers[0], input_dim=nx, activation=activations[0],
                             kernel_regularizer=K.regularizers.l2(lambtha)))
    model.add(K.layers.Dropout(1 - keep_prob))

    for i in range(1, len(layers)):
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(lambtha)))
        model.add(K.layers.Dropout(1 - keep_prob))

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model