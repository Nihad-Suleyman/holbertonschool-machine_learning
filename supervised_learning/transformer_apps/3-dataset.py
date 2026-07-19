#!/usr/bin/env python3
"""Prepare a data pipeline for Portuguese-English translation."""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Load, tokenize, filter, and batch a translation dataset."""

    def __init__(self, batch_size, max_len):
        """
        Initialize the translation data pipeline.

        Args:
            batch_size: Size of each training and validation batch.
            max_len: Maximum number of tokens allowed per sentence.
        """
        self.data_train = load_pt2en("train")
        self.data_valid = load_pt2en("validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_length(pt, en):
            """Keep pairs whose sentences do not exceed max_len."""
            pt_valid = tf.size(pt) <= max_len
            en_valid = tf.size(en) <= max_len
            return tf.logical_and(pt_valid, en_valid)

        self.data_train = self.data_train.filter(filter_max_length)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(
            batch_size,
            padded_shapes=([None], [None])
        )
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(
            batch_size,
            padded_shapes=([None], [None])
        )

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
        Encode Portuguese and English sentences.

        Args:
            pt: Tensor containing the Portuguese sentence.
            en: Tensor containing the English sentence.

        Returns:
            Lists of Portuguese and English token IDs.
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
        en_start = self.tokenizer_en.vocab_size

        pt_tokens = [pt_start] + pt_tokens + [pt_start + 1]
        en_tokens = [en_start] + en_tokens + [en_start + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Wrap encode for use inside a TensorFlow data pipeline.

        Args:
            pt: Tensor containing the Portuguese sentence.
            en: Tensor containing the English sentence.

        Returns:
            Encoded Portuguese and English tensors.
        """
        pt_tokens, en_tokens = tf.py_function(
            self.encode,
            [pt, en],
            [tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
