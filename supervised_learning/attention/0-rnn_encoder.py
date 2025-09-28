#!/usr/bin/env python3
"""
sadsadsadsadsa
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    sadasdsad
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        asdsadsa
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding
        )

        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform"
        )

    def initialize_hidden_state(self):
        """
        asdsadsadas
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        dsadsadsadsadsa
        """

        x = self.embedding(x)

        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
