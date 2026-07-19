#!/usr/bin/env python3
"""Prepare a dataset for Portuguese-to-English translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Load translation data and create sub-word tokenizers."""

    def __init__(self):
        """Initialize the datasets and tokenizers."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        tokenizers = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizers[0]
        self.tokenizer_en = tokenizers[1]

    def tokenize_dataset(self, data):
        """
        Create Portuguese and English tokenizers.

        Args:
            data: A dataset containing Portuguese-English sentence pairs.

        Returns:
            A tuple containing the Portuguese and English tokenizers.
        """
        pretrained_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        pretrained_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def portuguese_iterator():
            """Yield batches of Portuguese sentences."""
            for portuguese, _ in data.batch(1000):
                yield [
                    sentence.decode("utf-8")
                    for sentence in portuguese.numpy()
                ]

        def english_iterator():
            """Yield batches of English sentences."""
            for _, english in data.batch(1000):
                yield [
                    sentence.decode("utf-8")
                    for sentence in english.numpy()
                ]

        tokenizer_pt = pretrained_pt.train_new_from_iterator(
            portuguese_iterator(),
            vocab_size=2 ** 13
        )

        tokenizer_en = pretrained_en.train_new_from_iterator(
            english_iterator(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
