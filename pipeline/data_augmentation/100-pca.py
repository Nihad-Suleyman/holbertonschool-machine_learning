#!/usr/bin/env python3
"""PCA color augmentation"""
import tensorflow as tf


def pca_color(image, alphas):
    """Performs PCA color augmentation"""
    image = tf.cast(image, tf.float32)
    original_shape = image.shape
    pixels = tf.reshape(image, (-1, 3))
    mean = tf.reduce_mean(pixels, axis=0)
    centered = pixels - mean
    num_pixels = tf.cast(tf.shape(centered)[0], tf.float32)
    cov = tf.matmul(centered, centered, transpose_a=True) / num_pixels
    eigvals, eigvecs = tf.linalg.eigh(cov)
    idx = tf.argsort(eigvals, direction='DESCENDING')
    eigvals = tf.gather(eigvals, idx)
    eigvecs = tf.gather(eigvecs, idx, axis=1)
    alphas = tf.convert_to_tensor(alphas, dtype=tf.float32)
    delta = tf.matmul(
        eigvecs,
        tf.reshape(alphas * eigvals, (3, 1))
    )
    delta = tf.reshape(delta, (1, 3))
    augmented = pixels + delta
    augmented = tf.reshape(augmented, original_shape)

    return augmented
