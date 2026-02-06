#!/usr/bin/env python3
"""
afsfasfsafsafasfsa
"""
import numpy as np
import string

def bag_of_words(sentences, vocab=None):
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

    features = np.array(vocab)

    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    feature_to_index = {word: i for i, word in enumerate(features)}

    for i, tokens in enumerate(processed):
        for token in tokens:
            if token in feature_to_index:
                embeddings[i, feature_to_index[token]] += 1

    return embeddings, features
