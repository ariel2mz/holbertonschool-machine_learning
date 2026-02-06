#!/usr/bin/env python3
"""
asfasfsafsaf
"""
import numpy as np
import string
import math


def tf_idf(sentences, vocab=None):
    """
    asfasfsafas
    fasfasfsa
    """
    processed = []
    for sent in sentences:
        sent = sent.lower()
        sent = sent.replace("'s", "")
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        tokens = sent.split()
        processed.append(tokens)

    if vocab is None:
        all_words = []
        for tokens in processed:
            all_words.extend(tokens)
        vocab = sorted(set(all_words))

    features = list(vocab)
    s = len(sentences)
    f = len(features)

    tf = np.zeros((s, f), dtype=float)
    df = np.zeros(f, dtype=int)

    feature_to_index = {word: i for i, word in enumerate(features)}

    for i, tokens in enumerate(processed):
        for token in tokens:
            if token in feature_to_index:
                tf[i, feature_to_index[token]] += 1

    for j in range(f):
        for i in range(s):
            if tf[i, j] > 0:
                df[j] += 1

    idf = np.zeros(f)
    for j in range(f):
        if df[j] > 0:
            idf[j] = math.log(s / df[j])

    tf_idf_matrix = tf * idf
    for i in range(s):
        norm = np.linalg.norm(tf_idf_matrix[i])
        if norm > 0:
            tf_idf_matrix[i] /= norm

    return tf_idf_matrix, features
