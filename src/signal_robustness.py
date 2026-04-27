# src/signal_robustness.py

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def fetch_data(ticker, period="1y"):
    data = yf.Ticker(ticker).history(period=period)
    return data

def example_signal(data):
    # simple signal: moving average crossover
    data['SMA10'] = data['Close'].rolling(10).mean()
    data['SMA50'] = data['Close'].rolling(50).mean()
    data['Signal'] = np.where(data['SMA10'] > data['SMA50'], 1, -1)
    return data

def main():
    ticker = 'AAPL'
    data = fetch_data(ticker)
    data = example_signal(data)
    print(data[['Close', 'SMA10', 'SMA50', 'Signal']].tail())

if __name__ == "__main__":
    main()