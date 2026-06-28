# BTC Forecasting

This project uses the past 24 hours of BTC market data to predict the BTC
close price for the following hour.

## Preprocessing

The raw datasets contain one row per minute. The preprocessing script converts
those rows into hourly rows, scales the input features using only the training
set, and creates 24-hour sequences.

Example:

```bash
./preprocess_data.py coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv \
  bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv
```

## Training

The forecasting script trains an LSTM model with mean-squared error as the loss
function and uses `tf.data.Dataset` to feed data into the model.

```bash
./forecast_btc.py --data btc_preprocessed.npz --epochs 20
```
