#!/usr/bin/env python3
"""Dense block"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in
    Densely Connected Convolutional Networks
    """
    init = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        Y = K.layers.BatchNormalization(axis=3)(X)
        Y = K.layers.Activation('relu')(Y)
        Y = K.layers.Conv2D(filters=4 * growth_rate,
                            kernel_size=(1, 1),
                            padding='same',
                            kernel_initializer=init)(Y)

        Y = K.layers.BatchNormalization(axis=3)(Y)
        Y = K.layers.Activation('relu')(Y)
        Y = K.layers.Conv2D(filters=growth_rate,
                            kernel_size=(3, 3),
                            padding='same',
                            kernel_initializer=init)(Y)

        X = K.layers.Concatenate(axis=3)([X, Y])
        nb_filters += growth_rate

    return X, nb_filters
