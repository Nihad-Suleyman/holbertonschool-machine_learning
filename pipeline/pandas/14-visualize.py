#!/usr/bin/env python3
"""last task, we will generalize everything we have done so far"""


df.drop(columns=['Weighted_Price'])
df.rename(columns={'Timestamp': 'Date'})
df = df.to_datetime(df['Timestamp'], unit='s')
df = df.set_index('Date')
df['Close'] = df['Close'].ffill()
df['Open'] = df['Open'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['High'] = df['High'].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
df = df.loc['2017':]
df = df.resample("D").agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})
