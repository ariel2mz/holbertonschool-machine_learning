#!/usr/bin/env python3
"""
afsfasfsafsafasfsa
"""
import numpy as np
import string

def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words to use for analysis
               If None, all words within sentences should be used

    Returns:
        embeddings: numpy.ndarray of shape (s, f)
        features: list of the features used for embeddings
    """
    processed = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = sentence.replace("'s", "")
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        tokens = sentence.split()
        processed.append(tokens)

    if vocab is None:
        all_words = []
        for tokens in processed:
            all_words.extend(tokens)
        vocab = sorted(set(all_words))

    features = list(vocab)

    s = len(sentences)
    f = len(features)

    embeddings = np.zeros((s, f), dtype=int)
    feature_to_index = {word: i for i, word in enumerate(features)}

    for i, tokens in enumerate(processed):
        for token in tokens:
            if token in feature_to_index:
                embeddings[i, feature_to_index[token]] += 1

    return embeddings, features
