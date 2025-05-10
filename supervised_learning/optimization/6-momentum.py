#!/usr/bin/env python3
"""
no se amigo recien empiezo el codigo
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    alpha: learnin rate
    beta1: cuanta velocidad conserva
    """
    op = tf.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return op
