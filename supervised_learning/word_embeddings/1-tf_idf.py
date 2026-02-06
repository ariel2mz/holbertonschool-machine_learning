#!/usr/bin/env python3
"""
asfasfsafsaf
"""
import numpy as np
import string
import math


def tf_idf(sentences, vocab=None):
    """
    asfasfsafsa
    safasfsaf
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

    tf_matrix = np.zeros((s, f), dtype=float)
    df = np.zeros(f, dtype=int)

    feature_to_index = {word: i for i, word in enumerate(features)}

    for i, tokens in enumerate(processed):
        word_count = {}
        for token in tokens:
            if token in feature_to_index:
                word_count[token] = word_count.get(token, 0) + 1

        for word, count in word_count.items():
            tf_matrix[i, feature_to_index[word]] = count / len(tokens)

    for j in range(f):
        for i in range(s):
            if tf_matrix[i, j] > 0:
                df[j] += 1

    idf = np.zeros(f, dtype=float)
    for j in range(f):
        if df[j] > 0:
            idf[j] = math.log(s / df[j])

    tf_idf_matrix = np.zeros((s, f), dtype=float)
    for i in range(s):
        for j in range(f):
            tf_idf_matrix[i, j] = tf_matrix[i, j] * idf[j]

    return tf_idf_matrix, features
