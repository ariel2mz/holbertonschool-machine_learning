#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    sadsasadsa
    sadsadsa
    """
    regularizer = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0,
            mode='fan_avg'),
        kernel_regularizer=regularizer
    )

    return layer(prev)
