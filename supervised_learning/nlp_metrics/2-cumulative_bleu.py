#!/usr/bin/env python3
"""
Calculates the cumulative n-gram BLEU score for a sentence
"""

import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.

    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence
        n: size of the largest n-gram to use for evaluation

    Returns:
        The cumulative n-gram BLEU score
    """
    sentence_len = len(sentence)

    # Closest reference length
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda r: abs(r - sentence_len))

    # Brevity penalty
    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_len / sentence_len)

    precisions = []

    for gram_size in range(1, n + 1):
        sentence_ngrams = [
            tuple(sentence[i:i + gram_size])
            for i in range(sentence_len - gram_size + 1)
        ]

        matches = 0

        for ngram in set(sentence_ngrams):
            sentence_count = sentence_ngrams.count(ngram)

            max_ref_count = 0
            for ref in references:
                ref_ngrams = [
                    tuple(ref[i:i + gram_size])
                    for i in range(len(ref) - gram_size + 1)
                ]
                max_ref_count = max(max_ref_count,
                                    ref_ngrams.count(ngram))

            matches += min(sentence_count, max_ref_count)

        precision = matches / len(sentence_ngrams)
        precisions.append(precision)

    # All n-gram scores are weighted evenly
    score = bp * np.exp(np.sum(np.log(precisions)) / n)

    return score
