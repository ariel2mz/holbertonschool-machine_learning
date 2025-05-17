#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow with L2 regularization.

    Parameters:
    prev -- tensor containing the output of the previous layer
    n -- number of nodes the new layer should contain
    activation -- activation function to be used on the layer
    lambtha -- L2 regularization parameter

    Returns:
    The output of the new layer
    """
    l2_regularizer = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=l2_regularizer,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                                 mode='fan_in')
    )

    return layer(prev)
