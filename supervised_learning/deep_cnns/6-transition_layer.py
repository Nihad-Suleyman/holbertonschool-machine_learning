#!/usr/bin/env python3
"""Transition layer"""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    Densely Connected Convolutional Networks
    """
    init = K.initializers.he_normal(seed=0)
    filters = int(nb_filters * compression)

    Y = K.layers.BatchNormalization(axis=3)(X)
    Y = K.layers.Activation('relu')(Y)
    Y = K.layers.Conv2D(filters=filters,
                        kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer=init)(Y)
    Y = K.layers.AveragePooling2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  padding='same')(Y)

    return Y, filters
