#!/usr/bin/env python3
"""Convolutional Autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.
    """

    encoder_input = keras.Input(shape=input_dims)
    x = encoder_input

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)

    encoder = keras.Model(encoder_input, x)

    decoder_input = keras.Input(shape=latent_dims)
    x = decoder_input

    for f in reversed(filters):
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    x = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)

    decoder_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(decoder_input, decoder_output)

    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(auto_input, decoded)

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
