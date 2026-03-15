#!/usr/bin/env python3
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""

    inputs = tf.keras.Input(shape=(nx,))
    x = inputs

    for i in range(len(layers)):
        x = tf.keras.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=tf.keras.regularizers.l2(lambtha)
        )(x)

        if i < len(layers) - 1:
            x = tf.keras.layers.Dropout(rate=1 - keep_prob)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model