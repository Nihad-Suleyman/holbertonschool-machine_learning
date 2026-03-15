#!/usr/bin/env python3
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with Keras Sequential"""
    model = tf.keras.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(tf.keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=tf.keras.regularizers.L2(lambtha),
                input_shape=(nx,)
            ))
        else:
            model.add(tf.keras.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=tf.keras.regularizers.L2(lambtha)
            ))
        if i < len(layers) - 1:
            model.add(tf.keras.layers.Dropout(1 - keep_prob))

    return model