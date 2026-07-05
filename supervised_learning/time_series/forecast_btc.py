#!/usr/bin/env python3
"""Train an RNN to forecast the BTC close price for the next hour."""

import argparse
import json
import os

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


def evaluate_actual_prices(model, x, y_actual, target_mean, target_std):
    """Evaluate predictions after converting them back to USD prices."""
    predictions = model.predict(x, verbose=0).reshape(-1)
    predictions = predictions * target_std + target_mean
    errors = predictions - y_actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return mae, rmse, predictions


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
    parser.add_argument("--results-output", default="btc_forecast_results.json")
    parser.add_argument("--predictions-output", default="btc_forecast_predictions.npz")
    args = parser.parse_args()

    data = np.load(args.data)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]
    y_train_actual = data.get("y_train_actual", y_train)
    y_val_actual = data.get("y_val_actual", y_val)
    y_test_actual = data.get("y_test_actual", y_test)
    target_mean = float(data.get("target_mean", 0))
    target_std = float(data.get("target_std", 1))

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
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("Scaled test metrics:")
    scaled_metrics = model.evaluate(test_ds, verbose=2)
    train_mae, train_rmse, _ = evaluate_actual_prices(
        model, x_train, y_train_actual, target_mean, target_std
    )
    val_mae, val_rmse, _ = evaluate_actual_prices(
        model, x_val, y_val_actual, target_mean, target_std
    )
    test_mae, test_rmse, test_predictions = evaluate_actual_prices(
        model, x_test, y_test_actual, target_mean, target_std
    )
    print("Actual USD metrics:")
    print(f"Train MAE: {train_mae:.2f}, RMSE: {train_rmse:.2f}")
    print(f"Validation MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}")
    print(f"Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")

    os.makedirs(os.path.dirname(args.results_output) or ".", exist_ok=True)
    with open(args.results_output, "w", encoding="utf-8") as results_file:
        json.dump(
            {
                "scaled_test_metrics": [
                    float(metric) for metric in np.atleast_1d(scaled_metrics)
                ],
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "validation_mae": val_mae,
                "validation_rmse": val_rmse,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "history": {
                    key: [float(value) for value in values]
                    for key, values in history.history.items()
                },
            },
            results_file,
            indent=2,
        )
    np.savez_compressed(
        args.predictions_output,
        y_test_actual=y_test_actual,
        y_test_pred=test_predictions,
    )
    model.save(args.output)
    print(f"Saved model to {args.output}")
    print(f"Saved results to {args.results_output}")
    print(f"Saved predictions to {args.predictions_output}")


if __name__ == "__main__":
    main()
