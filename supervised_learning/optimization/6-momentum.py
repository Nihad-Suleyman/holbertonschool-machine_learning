#!/usr/bin/env python3
"""Creates the momentum optimization operation in TensorFlow"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
