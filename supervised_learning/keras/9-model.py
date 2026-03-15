#!/usr/bin/env python3
"""Save and load a Keras model"""
import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model"""
    network.save(filename)


def load_model(filename):
    """loads a saved model"""
    return K.models.load_model(filename)
