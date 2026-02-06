#!/usr/bin/env python3
"""
Calculates the n-gram BLEU score
"""
import math
from collections import Counter


def ngram(sequence, n):
    """
    Generate n-grams from a list of words
    """
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score
    """
    if len(sentence) < n:
        return 0.0

    sent_ngrams = Counter(ngram(sentence, n))

    max_ref_counts = {}
    for ref in references:
        ref_ngrams = Counter(ngram(ref, n))
        for ng in ref_ngrams:
            max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), ref_ngrams[ng])

    clipped = sum(min(sent_ngrams[ng],
                      max_ref_counts.get(ng, 0)) for ng in sent_ngrams)
    total_ngrams = sum(sent_ngrams.values())
    precision = clipped / total_ngrams if total_ngrams > 0 else 0
    sent_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    crlen = min(ref_lens, key=lambda rl: (abs(rl - sent_len), rl))

    bp = 1.0 if sent_len > crlen else math.exp(1 - crlen / sent_len)

    return bp * precision
