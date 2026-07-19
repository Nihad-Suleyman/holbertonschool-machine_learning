#!/usr/bin/env python3
"""Dataset preparation for Portuguese-to-English translation."""

import transformers
from setup import load_pt2en


class Dataset:
    """Loads translation datasets and creates sub-word tokenizers."""

    def __init__(self):
        """Initialize the training and validation datasets."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        tokenizers = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = tokenizers

    def tokenize_dataset(self, data):
        """
        Create Portuguese and English sub-word tokenizers.

        Args:
            data: Dataset containing Portuguese-English sentence pairs.

        Returns:
            The Portuguese and English tokenizers.
        """
        pretrained_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )
        pretrained_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def portuguese_sentences():
            """Yield Portuguese sentences from the dataset."""
            for portuguese, _ in data:
                yield portuguese.numpy().decode("utf-8")

        def english_sentences():
            """Yield English sentences from the dataset."""
            for _, english in data:
                yield english.numpy().decode("utf-8")

        tokenizer_pt = pretrained_pt.train_new_from_iterator(
            portuguese_sentences(),
            vocab_size=2 ** 13
        )

        tokenizer_en = pretrained_en.train_new_from_iterator(
            english_sentences(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
