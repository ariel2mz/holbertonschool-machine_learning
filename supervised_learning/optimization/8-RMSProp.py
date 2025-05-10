#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    # Set up the RMSProp optimizer
    optimizer = tf.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)
    return optimizer
