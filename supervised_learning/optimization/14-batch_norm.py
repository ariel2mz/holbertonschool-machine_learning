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

    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg')
    )(prev)

    gam = tf.Variable(initial_value=tf.ones([n]), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros([n]), trainable=True)
    eps = 1e-7

    mean, var = tf.nn.moments(dense, axes=[0])
    bn = tf.nn.batch_normalization(dense, mean, var, beta, gam, eps)

    return activation(batch_norm)
