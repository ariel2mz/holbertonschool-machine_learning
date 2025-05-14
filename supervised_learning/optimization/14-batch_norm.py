#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.

    Parameters:
    - prev: tensor, activated output of the previous layer
    - n: int, number of nodes in the layer to be created
    - activation: activation function to use after bn

    Returns:
    - tensor, activated output for the layer
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    den = tf.keras.layers.Dense(units=n, kernel_initializer=initializer)(prev)

    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,
        scale=True
    )(den, training=True)

    # Activation function
    return activation(batch_norm)
