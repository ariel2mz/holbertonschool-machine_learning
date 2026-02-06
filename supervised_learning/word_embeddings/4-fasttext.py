#!/usr/bin/env python3
"""
asfgasgas
"""
from gensim.models import FastText


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds and trains a gensim fastText model

    Args:
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: minimum number of occurrences of a word
        window: maximum distance between current and predicted word
        negative: size of negative sampling
        cbow: training type; True for CBOW, False for Skip-gram
        epochs: number of iterations to train over
        seed: seed for random number generator
        workers: number of worker threads

    Returns:
        The trained FastText model
    """
    sg = 0 if cbow else 1

    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        seed=seed,
        epochs=epochs
    )

    return model
