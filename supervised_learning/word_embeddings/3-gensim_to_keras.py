#!/usr/bin/env python3
"""
Converts a gensim Word2Vec model to a Keras Embedding layer
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained gensim Word2Vec model to a trainable Keras
    Embedding layer.

    Args:
        model: trained gensim Word2Vec model

    Returns:
        A trainable Keras Embedding layer
    """
    weights = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return embedding
