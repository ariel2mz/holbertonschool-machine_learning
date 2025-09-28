#!/usr/bin/env python3
"""
sadsadsadsadsa
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    asdsadsa
    """
    def __init__(self, units):
        """
        sadasdsadsa
        """
        super(SelfAttention, self).__init__()

        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        asdsasa
        """

        s_prev_expanded = tf.expand_dims(s_prev, axis=1)

        score_hidden = self.U(hidden_states)
        score_decoder = self.W(s_prev_expanded)
        score_combined = score_hidden + score_decoder
        score = tf.nn.tanh(score_combined)
        score = self.V(score)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
