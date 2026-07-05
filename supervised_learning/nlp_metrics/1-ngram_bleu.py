#!/usr/bin/env python3
"""
Calculates the n-gram BLEU score for a sentence
"""

import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence
        n: size of the n-gram to use for evaluation

    Returns:
        The n-gram BLEU score
    """
    sentence_len = len(sentence)

    # Find closest reference length
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda r: abs(r - sentence_len))

    # Brevity penalty
    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_len / sentence_len)

    # Generate n-grams for candidate sentence
    sentence_ngrams = [
        tuple(sentence[i:i + n])
        for i in range(sentence_len - n + 1)
    ]

    # Count clipped n-gram matches
    matches = 0

    for ngram in set(sentence_ngrams):
        sentence_count = sentence_ngrams.count(ngram)

        max_ref_count = 0
        for ref in references:
            ref_ngrams = [
                tuple(ref[i:i + n])
                for i in range(len(ref) - n + 1)
            ]
            max_ref_count = max(max_ref_count, ref_ngrams.count(ngram))

        matches += min(sentence_count, max_ref_count)

    precision = matches / len(sentence_ngrams)

    return bp * precision
