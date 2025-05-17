#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.
    Parameters:
    cost (tensor): The cost of the network without L2 regularization.
    model (tf.keras.Model): The Keras model that includes layers with L2.
    Returns:
    el fuckin costo tio
    """
    l2_loss = tf.zeros_like(cost)
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            if layer.kernel_regularizer is not None:
                l2_loss += layer.kernel_regularizer(layer.kernel)

    return cost + l2_loss
