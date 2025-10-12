#!/usr/bin/env python3
"""
asdadsddasda
"""
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    sadasdsaadsa
    """
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    corpus = []
    file_names = sorted(os.listdir(corpus_path))
    for filename in file_names:
        file_path = os.path.join(corpus_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                corpus.append(f.read())
    corpus_embeddings = embed(corpus)
    sentence_embedding = embed([sentence])
    similarity = np.inner(sentence_embedding, corpus_embeddings)[0]
    best_idx = np.argmax(similarity)

    return corpus[best_idx]
