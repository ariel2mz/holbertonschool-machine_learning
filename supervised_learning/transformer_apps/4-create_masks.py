#!/usr/bin/env python3
"""
fasfsafas
"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    aaaaaaa
    """

    emask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    emask = emask[:, tf.newaxis, tf.newaxis, :]

    dmask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    dmask = dmask[:, tf.newaxis, tf.newaxis, :]

    batch_size = tf.shape(target)[0]
    seq_len = tf.shape(target)[1]

    lamask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    lamask = tf.broadcast_to(lamask, (batch_size, 1, seq_len, seq_len))

    dtpmask = tf.cast(tf.math.equal(target, 0), tf.float32)
    dtpmask = dtpmask[:, tf.newaxis, tf.newaxis, :]

    cmask = tf.maximum(lamask, dtpmask)

    return emask, cmask, dmask
