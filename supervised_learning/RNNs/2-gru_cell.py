#!/usr/bin/env python3
"""Defines a gated recurrent unit cell."""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit."""

    def __init__(self, i, h, o):
        """Initialize the weights and biases of the GRU cell."""
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        combined = np.concatenate((h_prev, x_t), axis=1)

        z = 1 / (1 + np.exp(
            -(np.matmul(combined, self.Wz) + self.bz)
        ))
        r = 1 / (1 + np.exp(
            -(np.matmul(combined, self.Wr) + self.br)
        ))

        candidate_input = np.concatenate((r * h_prev, x_t), axis=1)
        h_intermediate = np.tanh(
            np.matmul(candidate_input, self.Wh) + self.bh
        )

        h_next = (1 - z) * h_prev + z * h_intermediate

        logits = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return h_next, y
