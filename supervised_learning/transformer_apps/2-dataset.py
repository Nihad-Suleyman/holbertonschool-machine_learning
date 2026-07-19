#!/usr/bin/env python3
"""Prepare a dataset for Portuguese-to-English translation."""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Load and tokenize a Portuguese-to-English translation dataset."""

    def __init__(self):
        """Initialize the datasets and their tokenizers."""
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        tokenizers = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = tokenizers[0]
        self.tokenizer_en = tokenizers[1]

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Create Portuguese and English sub-word tokenizers.

        Args:
            data: Dataset containing Portuguese-English sentence pairs.

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

    def encode(self, pt, en):
        """
        Encode Portuguese and English sentences into token lists.

        Args:
            pt: Tensor containing a Portuguese sentence.
            en: Tensor containing the corresponding English sentence.

        Returns:
            A tuple containing Portuguese and English token lists.
        """
        pt_sentence = pt.numpy().decode("utf-8")
        en_sentence = en.numpy().decode("utf-8")

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence,
            add_special_tokens=False
        )

        en_tokens = self.tokenizer_en.encode(
            en_sentence,
            add_special_tokens=False
        )

        pt_start = self.tokenizer_pt.vocab_size
        pt_end = pt_start + 1

        en_start = self.tokenizer_en.vocab_size
        en_end = en_start + 1

        pt_tokens = [pt_start] + pt_tokens + [pt_end]
        en_tokens = [en_start] + en_tokens + [en_end]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Wrap the encode method for use in a TensorFlow data pipeline.

        Args:
            pt: Tensor containing a Portuguese sentence.
            en: Tensor containing an English sentence.

        Returns:
            The encoded Portuguese and English tensors.
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
