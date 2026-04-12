#!/usr/bin/env python3
"""PCA color augmentation"""
import tensorflow as tf


def pca_color(image, alphas):
    """Performs PCA color augmentation"""
    img = tf.cast(image, tf.float32)
    shape = tf.shape(img)
    flat_img = tf.reshape(img, [-1, 3])
    mean = tf.reduce_mean(flat_img, axis=0)
    centered = flat_img - mean
    cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(centered)[0], tf.float32)
    eigvals, eigvecs = tf.linalg.eigh(cov)
    idx = tf.argsort(eigvals, direction='DESCENDING')
    eigvals = tf.gather(eigvals, idx)
    eigvecs = tf.gather(eigvecs, idx, axis=1)
    alphas = tf.convert_to_tensor(alphas, dtype=tf.float32)
    delta = tf.matmul(eigvecs, tf.expand_dims(alphas * eigvals, axis=1))
    delta = tf.squeeze(delta)
    flat_img = flat_img + delta
    augmented = tf.reshape(flat_img, shape)
    return augmented
