#!/usr/bin/env python3
"""
sadsdasdas
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    dasdasdasdsadas
    """

    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    sal = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        sal += (mask * -1e9)

    weights = tf.nn.softmax(sal, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
