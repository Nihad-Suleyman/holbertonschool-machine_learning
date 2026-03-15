#!/usr/bin/env python3
"""Converts a label vector to a one-hot matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """returns the one-hot matrix"""
    return K.utils.to_categorical(labels, num_classes=classes)