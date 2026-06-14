#!/usr/bin/env python3
"""Builds convolutional generator and discriminator models."""

from tensorflow import keras


def convolutional_GenDiscr():
    """Build a convolutional generator and discriminator.

    Returns:
        tuple: generator model and discriminator model
    """

    def get_generator():
        """Build the generator model."""

        inputs = keras.Input(shape=(16,))

        x = keras.layers.Dense(
            2 * 2 * 512,
            activation="tanh"
        )(inputs)

        x = keras.layers.Reshape(
            (2, 2, 512)
        )(x)

        x = keras.layers.UpSampling2D()(x)

        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            padding="same"
        )(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D()(x)

        x = keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding="same"
        )(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.UpSampling2D()(x)

        x = keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            padding="same"
        )(x)

        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation("tanh")(x)

        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="generator"
        )

    def get_discriminator():
        """Build the discriminator model."""

        inputs = keras.Input(shape=(16, 16, 1))

        x = keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same"
        )(inputs)

        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            padding="same"
        )(x)

        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            padding="same"
        )(x)

        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Conv2D(
            filters=256,
            kernel_size=3,
            padding="same"
        )(x)

        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Activation("tanh")(x)

        x = keras.layers.Flatten()(x)

        outputs = keras.layers.Dense(
            1,
            activation="tanh"
        )(x)

        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="discriminator"
        )

    return get_generator(), get_discriminator()
