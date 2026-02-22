#!/usr/bin/env python3
"""Creates a confusion matrix"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """a confusion matrix"""
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))
    true_classes = np.argmax(labels, axis=1)
    predicted_classes = np.argmax(logits, axis=1)
    for t, p in zip(true_classes, predicted_classes):
        confusion[t, p] += 1

    return confusion