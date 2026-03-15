#!/usr/bin/env python3
"""Makes predictions using a neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """makes a prediction using a neural network"""
    return network.predict(data, verbose=1 if verbose else 0)
