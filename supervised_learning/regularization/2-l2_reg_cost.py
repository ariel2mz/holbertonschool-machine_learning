#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a neural network with L2 regularization.
    
    Parameters:
    cost -- tensor containing the cost of the network without L2 regularization
    model -- Keras model that includes layers with L2 regularization
    
    Returns:
    el costo de cada layer con l2 aplciada 
    """
    l2_cost = tf.zeros_like(cost)
    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            l2 = layer.kernel_regularizer.l2
            weights = layer.kernel
            l2_cost += l2 * tf.nn.l2_loss(weights)
    total_cost = cost + l2_cost

    return total_cost
