#!/usr/bin/env python3
"""Creates a variational autoencoder."""

import tensorflow.keras as keras
import tensorflow.keras.backend as K


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a variational autoencoder.
    """

    def sampling(args):
        """
        Samples a latent vector using the reparameterization trick.
        """
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean))

        return mean + K.exp(log_var / 2) * epsilon

    encoder_input = keras.Input(shape=(input_dims,))
    encoded = encoder_input

    for nodes in hidden_layers:
        encoded = keras.layers.Dense(
            nodes,
            activation="relu"
        )(encoded)

    mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    log_var = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    latent = keras.layers.Lambda(
        sampling
    )([mean, log_var])

    encoder = keras.Model(
        inputs=encoder_input,
        outputs=[latent, mean, log_var]
    )

    decoder_input = keras.Input(shape=(latent_dims,))
    decoded = decoder_input

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(
            nodes,
            activation="relu"
        )(decoded)

    decoder_output = keras.layers.Dense(
        input_dims,
        activation="sigmoid"
    )(decoded)

    decoder = keras.Model(
        inputs=decoder_input,
        outputs=decoder_output
    )

    auto_output = decoder(latent)

    auto = keras.Model(
        inputs=encoder_input,
        outputs=auto_output
    )

    kl_loss = -0.5 * K.sum(
        1 + log_var - K.square(mean) - K.exp(log_var),
        axis=-1
    )

    auto.add_loss(K.mean(kl_loss))

    auto.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    return encoder, decoder, auto