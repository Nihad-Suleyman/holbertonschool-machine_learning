#!/usr/bin/env python3
"""PCA color augmentation"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """Performs PCA color augmentation"""
    img = tf.cast(image, tf.float32)
    shape = img.shape
    flat_img = tf.reshape(img, [-1, 3])
    mean = tf.reduce_mean(flat_img, axis=0)
    centered = flat_img - mean
    cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(centered)[0], tf.float32)
    cov_np = cov.numpy()
    eigvals, eigvecs = np.linalg.eigh(cov_np)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    alphas = np.array(alphas)
    delta = eigvecs @ (alphas * eigvals)
    flat_img = flat_img + delta
    augmented = tf.reshape(flat_img, shape)

    return augmented
