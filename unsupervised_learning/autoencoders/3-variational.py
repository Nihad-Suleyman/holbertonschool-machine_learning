#!/usr/bin/env python3
"""Variational Autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder"""

    def sampling(args):
        """Reparameterization trick"""
        z_mean, z_log_sigma = args

        batch = keras.backend.shape(z_mean)[0]
        dims = keras.backend.int_shape(z_mean)[1]

        epsilon = keras.backend.random_normal(
            shape=(batch, dims)
        )

        return z_mean + keras.backend.exp(
            z_log_sigma / 2
        ) * epsilon

    # Encoder
    inputs = keras.Input(shape=(input_dims,))

    encoded = inputs
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(
            nodes,
            activation='relu'
        )(encoded)

    z_mean = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    z_log_sigma = keras.layers.Dense(
        latent_dims,
        activation=None
    )(encoded)

    z = keras.layers.Lambda(
        sampling
    )([z_mean, z_log_sigma])

    encoder = keras.Model(
        inputs,
        [z, z_mean, z_log_sigma]
    )

    # Decoder
    latent_inputs = keras.Input(
        shape=(latent_dims,)
    )

    decoded = latent_inputs

    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(
            nodes,
            activation='relu'
        )(decoded)

    outputs = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(decoded)

    decoder = keras.Model(
        latent_inputs,
        outputs
    )

    # Autoencoder
    z, z_mean, z_log_sigma = encoder(inputs)

    vae_outputs = decoder(z)

    auto = keras.Model(
        inputs,
        vae_outputs
    )

    # KL divergence loss
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_log_sigma
        - keras.backend.square(z_mean)
        - keras.backend.exp(z_log_sigma),
        axis=-1
    )

    auto.add_loss(
        keras.backend.mean(kl_loss)
    )

    auto.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder, decoder, auto
