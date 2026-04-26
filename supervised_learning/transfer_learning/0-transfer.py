#!/usr/bin/env python3
"""
Transfer learning on CIFAR-10
"""

import tensorflow.keras as K
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np


def preprocess_data(X, Y):
    """
    Preprocesses CIFAR-10 data for MobileNetV2.
    """
    X_p = preprocess_input(X.astype("float32"))
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


def build_model():
    """
    Builds transfer learning model.
    """
    inputs = K.Input(shape=(32, 32, 3))

    x = K.layers.Resizing(160, 160)(inputs)

    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(160, 160, 3)
    )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation="relu")(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Dropout(0.4)(x)
    outputs = K.layers.Dense(10, activation="softmax")(x)

    model = K.Model(inputs, outputs)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)

    model = build_model()

    callbacks = [
        K.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),
        K.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    model.fit(
        X_train,
        Y_train,
        validation_data=(X_valid, Y_valid),
        batch_size=64,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    model.save("cifar10.h5")
