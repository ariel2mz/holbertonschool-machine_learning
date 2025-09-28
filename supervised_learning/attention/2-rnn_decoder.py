#!/usr/bin/env python3
"""
sadsadasdsaa
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    sadasdsadsa
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        sadsadsadsa
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)

        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        sadsadsad
        """
        cv, aw = self.attention(s_prev, hidden_states)

        x_embed = self.embedding(x)

        cv = tf.expand_dims(cv, axis=1)

        decoder_input = tf.concat([cv, x_embed], axis=-1)

        output, s = self.gru(decoder_input, initial_state=s_prev)

        output = tf.reshape(output, (-1, output.shape[2]))

        y = self.F(output)
        return y, s
