#!/usr/bin/env python3
"""PCA color augmentation"""
import tensorflow as tf


def pca_color(image, alphas):
    """Performs PCA color augmentation"""

    image = tf.cast(image, tf.float32)
    eigvals = tf.constant([0.2175, 0.0188, 0.0045], dtype=tf.float32)

    eigvecs = tf.constant([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203]
    ], dtype=tf.float32)

    alphas = tf.convert_to_tensor(alphas, dtype=tf.float32)

    delta = tf.matmul(
        eigvecs,
        tf.reshape(alphas * eigvals, (3, 1))
    )

    delta = tf.reshape(delta, (1, 1, 3))

    return image + delta
