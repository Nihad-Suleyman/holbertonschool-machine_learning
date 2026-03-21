#!/usr/bin/env python3
"""L2 Regularization Cost with Keras"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization
    """
    return cost + tf.stack(model.losses)
