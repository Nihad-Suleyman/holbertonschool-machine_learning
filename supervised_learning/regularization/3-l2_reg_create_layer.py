#!/usr/bin/env python3
"""Create a layer with L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a tensorflow layer that includes L2 regularization
    """
    regularizer = tf.keras.regularizers.l2(lambtha)
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode="fan_avg"
        ),
        kernel_regularizer=regularizer
    )
    return layer(prev)
