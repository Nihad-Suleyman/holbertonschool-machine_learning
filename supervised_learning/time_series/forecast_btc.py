#!/usr/bin/env python3
"""Train an RNN to forecast the BTC close price for the next hour."""

import argparse

import numpy as np
import tensorflow as tf


def make_dataset(x, y, batch_size=64, shuffle=False):
    """Build a tf.data.Dataset from numpy arrays."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(min(len(x), 10000), seed=42)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(window_size, feature_count):
    """Build a simple LSTM regression model."""
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, feature_count)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])


def main():
    """Load preprocessed BTC data and train the forecasting model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--data", default="btc_preprocessed.npz",
        help="Path to the .npz file created by preprocess_data.py"
    )
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-o", "--output", default="btc_forecast_model.keras")
    args = parser.parse_args()

    data = np.load(args.data)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    train_ds = make_dataset(x_train, y_train, args.batch_size, shuffle=True)
    val_ds = make_dataset(x_val, y_val, args.batch_size)
    test_ds = make_dataset(x_test, y_test, args.batch_size)

    model = build_model(x_train.shape[1], x_train.shape[2])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.summary()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("Test metrics:")
    model.evaluate(test_ds, verbose=2)
    model.save(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
