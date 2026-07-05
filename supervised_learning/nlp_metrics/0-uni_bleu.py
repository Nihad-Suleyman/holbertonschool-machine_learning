#!/usr/bin/env python3
"""
Calculates the unigram BLEU score for a sentence
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.

    Args:
        references: list of reference translations
        sentence: list containing the model proposed sentence

    Returns:
        The unigram BLEU score
    """
    sentence_len = len(sentence)

    # Find the reference length closest to the sentence length
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda r: abs(r - sentence_len))

    # Brevity penalty
    if sentence_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_len / sentence_len)

    # Count clipped unigram matches
    matches = 0

    for word in set(sentence):
        sentence_count = sentence.count(word)
        max_ref_count = max(ref.count(word) for ref in references)
        matches += min(sentence_count, max_ref_count)

    precision = matches / sentence_len

    return bp * precision
