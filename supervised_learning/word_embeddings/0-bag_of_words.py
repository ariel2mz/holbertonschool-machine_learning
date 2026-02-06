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
        vocab = []
        seen = set()
        for tokens in processed:
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    vocab.append(token)

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
