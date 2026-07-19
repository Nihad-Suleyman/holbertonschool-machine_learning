#!/usr/bin/env python3
"""Loads and prepares a Portuguese-to-English translation dataset."""

from transformers import AutoTokenizer
from setup import load_pt2en


class Dataset:
    """Loads a dataset and creates Portuguese and English tokenizers."""

    def __init__(self):
        """Initialize the training data, validation data, and tokenizers."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Create Portuguese and English sub-word tokenizers.

        Args:
            data: A tf.data.Dataset containing (Portuguese, English) pairs.

        Returns:
            tokenizer_pt: The trained Portuguese tokenizer.
            tokenizer_en: The trained English tokenizer.
        """
        base_tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

        base_tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

        def portuguese_iterator():
            """Yield Portuguese sentences as strings."""
            for pt, _ in data.as_numpy_iterator():
                yield pt.decode("utf-8")

        def english_iterator():
            """Yield English sentences as strings."""
            for _, en in data.as_numpy_iterator():
                yield en.decode("utf-8")

        tokenizer_pt = base_tokenizer_pt.train_new_from_iterator(
            portuguese_iterator(),
            vocab_size=2 ** 13
        )

        tokenizer_en = base_tokenizer_en.train_new_from_iterator(
            english_iterator(),
            vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
