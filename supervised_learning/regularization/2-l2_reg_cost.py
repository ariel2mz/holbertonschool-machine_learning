#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a neural network with L2 regularization.
    Parameters:
    cost (tensor): The cost of the network without L2 regularization.
    model (tf.keras.Model): The Keras model that includes layers with L2.
    Returns:
    tensor: A tensor containing the total cost for each layer of the network,
            accounting for L2 regularization.
    """
    total_l2_costs = []
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            layer_l2_cost = tf.reduce_sum(layer.losses)
            total_cost = layer_l2_cost + cost
            total_l2_costs.append(total_cost)
    """
    basicamente en vez de sumar todo, suma una layer y
    lo apendea entonces quedan varios valores
    """
    return tf.convert_to_tensor(total_l2_costs)
