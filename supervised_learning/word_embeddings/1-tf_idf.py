#!/usr/bin/env python3
"""
asfasfsafsaf
"""
import numpy as np
import string
import math


def tf_idf(sentences, vocab=None):
    processed = []
    for sent in sentences:
        sent = sent.lower()
        sent = sent.replace("'s", "")
        sent = sent.translate(str.maketrans('', '', string.punctuation))
        processed.append(sent.split())

    if vocab is None:
        all_words = []
        for tokens in processed:
            all_words.extend(tokens)
        vocab = sorted(set(all_words))

    features = list(vocab)
    s = len(sentences)
    f = len(features)

    tf = np.zeros((s, f), dtype=float)
    feature_to_index = {word: i for i, word in enumerate(features)}

    for i, tokens in enumerate(processed):
        for token in tokens:
            if token in feature_to_index:
                tf[i, feature_to_index[token]] += 1

    df = np.zeros(f, dtype=float)
    for j in range(f):
        df[j] = np.count_nonzero(tf[:, j])

    idf = 1.0 + np.log((1.0 + s) / (1.0 + df))

    tf_idf_matrix = tf * idf

    norms = np.linalg.norm(tf_idf_matrix, axis=1)
    for i in range(s):
        if norms[i] != 0:
            tf_idf_matrix[i] /= norms[i]

    return tf_idf_matrix, features

