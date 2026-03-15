#!/usr/bin/env python3
"""Save and load model configuration"""
import tensorflow.keras as K


def save_config(network, filename):
    """saves a model's configuration in JSON format"""
    config = network.to_json()

    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """loads a model with a specific configuration"""
    with open(filename, 'r') as f:
        config = f.read()

    model = K.models.model_from_json(config)

    return model
