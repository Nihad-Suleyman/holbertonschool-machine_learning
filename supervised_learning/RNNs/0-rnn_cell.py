#!/usr/bin/env python3
"""Defines a simple recurrent neural network cell."""

import numpy as np


class RNNCell:
    """Represents a cell of a simple recurrent neural network."""

    def __init__(self, i, h, o):
        """Initialize the weights and biases of the RNN cell."""
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step."""
        combined = np.concatenate((h_prev, x_t), axis=1)

        h_next = np.tanh(np.matmul(combined, self.Wh) + self.bh)

        logits = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return h_next, y
