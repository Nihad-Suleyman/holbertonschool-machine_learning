#!/usr/bin/env python3
"""Performs a convolution on images with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on images with channels
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil((((h - 1) * sh + kh - h) / 2)))
        pw = int(np.ceil((((w - 1) * sw + kw - w) / 2)))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )
    output_h = ((h + 2 * ph - kh) // sh) + 1
    output_w = ((w + 2 * pw - kw) // sw) + 1
    output = np.zeros((m, output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            output[:, i, j] = np.sum(
                padded[:, h_start:h_end, w_start:w_end, :] * kernel,
                axis=(1, 2, 3)
            )

    return output
