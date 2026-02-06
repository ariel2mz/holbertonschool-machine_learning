#!/usr/bin/env python3
"""
asfgasgas
"""
from tensorflow.keras.layers import Embedding
import numpy as np


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

    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )

    return embedding_layer
