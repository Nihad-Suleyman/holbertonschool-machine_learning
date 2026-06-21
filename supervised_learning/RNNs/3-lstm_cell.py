#!/usr/bin/env python3
"""Defines a long short-term memory cell."""

import numpy as np


class LSTMCell:
    """Represents a long short-term memory unit."""

    def __init__(self, i, h, o):
        """Initialize the weights and biases of the LSTM cell."""
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step."""
        combined = np.concatenate((h_prev, x_t), axis=1)

        forget = 1 / (1 + np.exp(
            -(np.matmul(combined, self.Wf) + self.bf)
        ))
        update = 1 / (1 + np.exp(
            -(np.matmul(combined, self.Wu) + self.bu)
        ))
        candidate = np.tanh(
            np.matmul(combined, self.Wc) + self.bc
        )
        output = 1 / (1 + np.exp(
            -(np.matmul(combined, self.Wo) + self.bo)
        ))

        c_next = forget * c_prev + update * candidate
        h_next = output * np.tanh(c_next)

        logits = np.matmul(h_next, self.Wy) + self.by
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        y = exp / np.sum(exp, axis=1, keepdims=True)

        return h_next, c_next, y
