#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    alpha: el learning rate
    beta2: el peso q se descuenta o algo asi
    epsilon: numero chiquito

    returns: op
    """
    op = tf.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)
    return op
