#!/usr/bin/env python3
"""Preprocess raw BTC minute data for hourly forecasting."""

import argparse
import json
import os

import numpy as np
import pandas as pd


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price",
]


def load_exchange(path):
    """Load one exchange CSV and return cleaned minute-level data."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["Timestamp", "Close"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")
    return df.set_index("Timestamp").sort_index()


def to_hourly(df):
    """Convert minute-level rows into one row per hour."""
    hourly = pd.DataFrame()
    hourly["Open"] = df["Open"].resample("1H").first()
    hourly["High"] = df["High"].resample("1H").max()
    hourly["Low"] = df["Low"].resample("1H").min()
    hourly["Close"] = df["Close"].resample("1H").last()
    hourly["Volume_(BTC)"] = df["Volume_(BTC)"].resample("1H").sum()
    hourly["Volume_(Currency)"] = df["Volume_(Currency)"].resample("1H").sum()
    hourly["Weighted_Price"] = df["Weighted_Price"].resample("1H").mean()
    return hourly.dropna()


def make_sequences(values, targets, window_size=24):
    """Create X/y pairs where X is the previous 24 hours."""
    x, y = [], []
    for idx in range(window_size, len(values)):
        x.append(values[idx - window_size:idx])
        y.append(targets[idx])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    """Create train/validation/test arrays from raw Coinbase/Bitstamp CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_files", nargs="+", help="Raw BTC CSV files")
    parser.add_argument("-o", "--output", default="btc_preprocessed.npz")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    args = parser.parse_args()

    hourly_frames = [to_hourly(load_exchange(path)) for path in args.csv_files]
    data = pd.concat(hourly_frames).sort_index()
    data = data.groupby(data.index).mean()
    data = data[FEATURES].dropna()

    train_end = int(len(data) * (1 - args.validation_split - args.test_split))
    val_end = int(len(data) * (1 - args.test_split))

    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end - args.window_size:val_end]
    test_data = data.iloc[val_end - args.window_size:]

    mean = train_data.mean()
    std = train_data.std().replace(0, 1)

    def scale(frame):
        return ((frame - mean) / std).to_numpy(dtype=np.float32)

    x_train, y_train = make_sequences(
        scale(train_data), train_data["Close"].to_numpy(dtype=np.float32),
        args.window_size
    )
    x_val, y_val = make_sequences(
        scale(val_data), val_data["Close"].to_numpy(dtype=np.float32),
        args.window_size
    )
    x_test, y_test = make_sequences(
        scale(test_data), test_data["Close"].to_numpy(dtype=np.float32),
        args.window_size
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        features=np.array(FEATURES),
    )

    scaler_path = os.path.splitext(args.output)[0] + "_scaler.json"
    with open(scaler_path, "w", encoding="utf-8") as scaler_file:
        json.dump(
            {
                "features": FEATURES,
                "mean": mean.to_dict(),
                "std": std.to_dict(),
                "window_size": args.window_size,
            },
            scaler_file,
            indent=2,
        )

    print(f"Saved arrays to {args.output}")
    print(f"Saved scaler data to {scaler_path}")
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")


if __name__ == "__main__":
    main()
