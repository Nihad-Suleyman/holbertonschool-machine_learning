#!/usr/bin/env python3
"""Performs a convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant'
    )
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )

    return output
