#!/usr/bin/env python3

import tensorflow.keras as K
"""
This module builds a neural network using the Functional API in Keras.
"""


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network model using the Keras Functional API.

    Parameters:
    - nx: number of input features
    - layers: list containing the number of nodes in each layer
    - activations: list containing activation functions for each layer
    - lambtha: L2 regularization parameter
    - keep_prob: probability of keeping a node during dropout

    Returns:
    - A compiled Keras model.
    """
    # Input layer
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha)
    )(inputs)

    # Hidden layers with Dropout
    for i in range(1, len(layers)):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

    # Create the model with the explicit name
    model = K.Model(inputs=inputs, outputs=x, name="model")

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
