#!/usr/bin/env python3
"""
asfgasgas
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a keras Embedding layer

    Args:
        model: trained gensim word2vec model

    Returns:
        Trainable keras Embedding layer
    """
    weights = model.wv.vectors
    vocab_size, embedding_dim = weights.shape

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
