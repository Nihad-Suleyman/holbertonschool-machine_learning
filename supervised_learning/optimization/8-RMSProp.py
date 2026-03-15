#!/usr/bin/env python3
"""Creates the RMSProp optimization operation in TensorFlow"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
