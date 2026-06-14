#!/usr/bin/env python3
"""Defines a simple Generative Adversarial Network."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class Simple_GAN(keras.Model):
    """Simple Generative Adversarial Network."""

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=.005
    ):
        """Initialize the Simple GAN."""

        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5
        self.beta_2 = .9

        # Generator loss and optimizer
        self.generator.loss = lambda x: (
            tf.keras.losses.MeanSquaredError()(
                x,
                tf.ones(x.shape)
            )
        )

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # Discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(
                x,
                tf.ones(x.shape)
            )
            +
            tf.keras.losses.MeanSquaredError()(
                y,
                -tf.ones(y.shape)
            )
        )

        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """Generate fake samples."""

        if size is None:
            size = self.batch_size

        latent_sample = self.latent_generator(size)

        return self.generator(
            latent_sample,
            training=training
        )

    def get_real_sample(self, size=None):
        """Select random real samples."""

        if size is None:
            size = self.batch_size

        sorted_indices = tf.range(
            tf.shape(self.real_examples)[0]
        )

        random_indices = tf.random.shuffle(
            sorted_indices
        )[:size]

        return tf.gather(
            self.real_examples,
            random_indices
        )

    def train_step(self, useless_argument):
        """Perform one GAN training step."""

        # Train the discriminator disc_iter times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()

                # We do not train the generator during this stage
                fake_sample = self.get_fake_sample(
                    training=False
                )

                real_prediction = self.discriminator(
                    real_sample,
                    training=True
                )

                fake_prediction = self.discriminator(
                    fake_sample,
                    training=True
                )

                discr_loss = self.discriminator.loss(
                    real_prediction,
                    fake_prediction
                )

            discriminator_gradients = tape.gradient(
                discr_loss,
                self.discriminator.trainable_variables
            )

            self.discriminator.optimizer.apply_gradients(
                zip(
                    discriminator_gradients,
                    self.discriminator.trainable_variables
                )
            )

        # Train the generator once
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(
                training=True
            )

            fake_prediction = self.discriminator(
                fake_sample,
                training=False
            )

            gen_loss = self.generator.loss(
                fake_prediction
            )

        generator_gradients = tape.gradient(
            gen_loss,
            self.generator.trainable_variables
        )

        self.generator.optimizer.apply_gradients(
            zip(
                generator_gradients,
                self.generator.trainable_variables
            )
        )

        return {
            "discr_loss": discr_loss,
            "gen_loss": gen_loss
        }
