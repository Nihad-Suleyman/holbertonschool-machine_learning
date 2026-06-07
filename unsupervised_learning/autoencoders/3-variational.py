#!/usr/bin/env python3
"""Variational Autoencoder"""

import tensorflow.keras as keras
from tensorflow.keras import backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.

    Returns:
        encoder, decoder, auto
    """

    def sampling(args):
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean))
        return mean + K.exp(log_var / 2) * epsilon

    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    z = keras.layers.Lambda(sampling)([mean, log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[z, mean, log_var]
    )

    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    z, mean, log_var = encoder(encoder_input)
    reconstructed = decoder(z)

    auto = keras.Model(
    inputs=encoder_input,
    outputs=reconstructed
    )

    auto.compile(
        optimizer='adam',
        loss=keras.losses.binary_crossentropy
    )

    return encoder, decoder, auto
