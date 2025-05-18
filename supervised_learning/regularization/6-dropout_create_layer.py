#!/usr/bin/env python3
"""
sadasdasdsad
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function for the new layer
        keep_prob: probability that a node will be kept
        training: boolean indicating whether the model is in training mode

    Returns:
        Output of the new layer
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    retorno = dense_layer(prev)

    # aca esta el fukin dropout
    if training and keep_prob < 1:
        retorno = tf.nn.dropout(retorno, rate=1 - keep_prob)

    return retorno
