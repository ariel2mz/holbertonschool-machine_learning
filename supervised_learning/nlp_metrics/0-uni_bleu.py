#!/usr/bin/env python3
"""
Calculates the unigram BLEU score
"""
import math
from collections import Counter


def uni_bleu(references, sentence):
    """
    Calculates unigram BLEU score
    """

    sent_count = Counter(sentence)
    max_ref_counts = {}

    for ref in references:
        ref_count = Counter(ref)
        for word in ref_count:
            max_ref_counts[word] = max(
                max_ref_counts.get(word, 0),
                ref_count[word]
            )
    clipped = 0
    for word in sent_count:
        clipped += min(sent_count[word], max_ref_counts.get(word, 0))
    precision = clipped / len(sentence) if sentence else 0
    sent_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda rl: (abs(rl - sent_len), rl))

    if sent_len > closest_ref_len:
        bp = 1
    else:
        bp = math.exp(1 - closest_ref_len / sent_len)

    return bp * precision
