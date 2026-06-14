#!/usr/bin/env python3
"""Defines a Wasserstein GAN using discriminator weight clipping."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    """Wasserstein GAN with discriminator weight clipping."""

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
        """Initialize the Wasserstein GAN."""

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

        # Generator minimizes the negative average discriminator score
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)

        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2
        )

        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss
        )

        # x: discriminator output for real examples
        # y: discriminator output for fake examples
        self.discriminator.loss = lambda x, y: (
            tf.math.reduce_mean(y) - tf.math.reduce_mean(x)
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
        """Generate fake samples from latent vectors."""

        if size is None:
            size = self.batch_size

        latent_sample = self.latent_generator(size)

        return self.generator(
            latent_sample,
            training=training
        )

    def get_real_sample(self, size=None):
        """Return a random batch of real examples."""

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
        """Perform one Wasserstein GAN training step."""

        # Train the discriminator multiple times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()

                fake_sample = self.get_fake_sample(
                    training=False
                )

                real_output = self.discriminator(
                    real_sample,
                    training=True
                )

                fake_output = self.discriminator(
                    fake_sample,
                    training=True
                )

                discr_loss = self.discriminator.loss(
                    real_output,
                    fake_output
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

            # Clip every discriminator weight and bias to [-1, 1]
            for variable in self.discriminator.trainable_variables:
                variable.assign(
                    tf.clip_by_value(variable, -1.0, 1.0)
                )

        # Train the generator once
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(
                training=True
            )

            fake_output = self.discriminator(
                fake_sample,
                training=False
            )

            gen_loss = self.generator.loss(fake_output)

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
