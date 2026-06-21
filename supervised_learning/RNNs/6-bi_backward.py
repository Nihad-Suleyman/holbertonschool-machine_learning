#!/usr/bin/env python3
"""Defines forward and backward operations of a bidirectional RNN."""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional recurrent neural network cell."""

    def __init__(self, i, h, o):
        """Initialize the cell's weights and biases."""
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculate the next forward hidden state."""
        combined = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(
            np.matmul(combined, self.Whf) + self.bhf
        )

        return h_next

    def backward(self, h_next, x_t):
        """Calculate the previous backward hidden state."""
        combined = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(
            np.matmul(combined, self.Whb) + self.bhb
        )

        return h_prev
