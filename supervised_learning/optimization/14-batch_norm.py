#!/usr/bin/env python3
"""Creates a batch normalization layer for a neural network in TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    x = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )(prev)

    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        beta_initializer='zeros',
        gamma_initializer='ones'
    )(x, training=True)

    return activation(x)