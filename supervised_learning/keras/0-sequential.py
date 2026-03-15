#!/usr/bin/env python3
""" 0-sequential.py """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ build model """

    par = [K.Input(shape=(nx,))]
    for i in range(len(layers)):
        par.append(K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        ))
        if i != len(layers) - 1:
            par.append(K.layers.Dropout(1 - keep_prob))
    model = K.Sequential(par)
    return model