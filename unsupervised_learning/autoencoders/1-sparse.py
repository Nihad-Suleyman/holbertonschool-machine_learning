#!/usr/bin/env python3
"""Sparse Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder

    Args:
        input_dims: dimensions of input
        hidden_layers: list containing hidden layer sizes
        latent_dims: dimensions of latent space
        lambtha: L1 regularization parameter

    Returns:
        encoder, decoder, auto
    """

    # Encoder
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    latent = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)

    encoder = keras.Model(encoder_input, latent)

    # Decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(x)

    decoder = keras.Model(decoder_input, decoder_output)

    # Autoencoder
    auto_input = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(auto_input, decoded)

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
