import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ----------------------------
# DATA
# ----------------------------

def fetch_data(ticker, period="3y"):
    data = yf.Ticker(ticker).history(period=period)
    return data

# ----------------------------
# SIGNAL
# ----------------------------

def generate_signal(data, short_window=10, long_window=50):
    df = data.copy()
    df["SMA_Short"] = df["Close"].rolling(short_window).mean()
    df["SMA_Long"] = df["Close"].rolling(long_window).mean()
    df["Signal"] = np.where(df["SMA_Short"] > df["SMA_Long"],1,-1)

    return df

# ----------------------------
# BACKTEST
# ----------------------------

def backtest(df):
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()

    # Use yesterday's signal
    df["Strategy_Returns"] = (df["Signal"].shift(1) * df["Returns"])
    return df

# ----------------------------
# SHARPE RATIO
# ----------------------------

def sharpe_ratio(returns):
    returns = returns.dropna()
    if len(returns) == 0:
        return 0

    std = returns.std()

    if std == 0:
        return 0

    return np.sqrt(252) * returns.mean() / std


# ----------------------------
# BOOTSTRAP ANALYSIS
# ----------------------------

def bootstrap_sharpe(strategy_returns, n_iterations=1000):
    strategy_returns = strategy_returns.dropna()
    bootstrap_results = []

    for _ in range(n_iterations):
        sample = np.random.choice(strategy_returns, size=len(strategy_returns),replace=True)
        sr = sharpe_ratio(pd.Series(sample))
        bootstrap_results.append(sr)

    return np.array(bootstrap_results)

# ----------------------------
# PARAMETER SENSITIVITY
# ----------------------------

def parameter_sensitivity(data):
    short_windows = [5, 10, 20]
    long_windows = [30, 50, 100]
    results = []

    for short in short_windows:
        for long in long_windows:
            if short >= long:
                continue

            df = generate_signal(data,short_window=short,long_window=long)
            df = backtest(df)
            sr = sharpe_ratio(df["Strategy_Returns"])

            results.append({"Short": short,"Long": long,"Sharpe": sr})

    return pd.DataFrame(results)


# ----------------------------
# TRAIN TEST VALIDATION
# ----------------------------

def train_test_validation(data, short_window=10, long_window=50):
    split = int(len(data) * 0.7)
    train = data.iloc[:split]
    test = data.iloc[split:]

    train = generate_signal(train,short_window,long_window)
    train = backtest(train)

    test = generate_signal(test,short_window,long_window)
    test = backtest(test)

    train_sharpe = sharpe_ratio(train["Strategy_Returns"])
    test_sharpe = sharpe_ratio(test["Strategy_Returns"])

    return train_sharpe, test_sharpe


# ----------------------------
# ROBUSTNESS SCORE
# ----------------------------

def robustness_score(actual_sharpe, bootstrap_sharpes, train_sharpe, test_sharpe, sensitivity_df):
    bootstrap_score = min(100, max(0, 50 + 25 * (actual_sharpe - bootstrap_sharpes.mean())))
    stability_gap = abs(train_sharpe - test_sharpe)
    stability_score = max(0, 100 - 50 * stability_gap)
    sensitivity_score = max(0, 100 - 50 * sensitivity_df["Sharpe"].std())

    final_score = (0.4 * bootstrap_score + 0.3 * stability_score + 0.3 * sensitivity_score)

    return round(final_score, 2)


# ----------------------------
# MAIN
# ----------------------------

def main():
    ticker = "AAPL"
    print(f"\nLoading {ticker} data...")
    data = fetch_data(ticker)

    # Base strategy
    data = generate_signal(data,short_window=10,long_window=50)
    data = backtest(data)
    actual_sharpe = sharpe_ratio(data["Strategy_Returns"])

    # Bootstrap
    bootstrap_sharpes = bootstrap_sharpe(data["Strategy_Returns"])

    # Sensitivity
    sensitivity_df = parameter_sensitivity(data)

    # Train/Test
    train_sharpe, test_sharpe = (train_test_validation(data))

    # Robustness Score
    score = robustness_score(actual_sharpe,bootstrap_sharpes,train_sharpe,test_sharpe,sensitivity_df)

    # Report
    print("\nSignal Robustness Report")
    print("-" * 35)

    print(f"Actual Sharpe: "f"{actual_sharpe:.2f}")
    print(f"Bootstrap Mean Sharpe: "f"{bootstrap_sharpes.mean():.2f}")
    print(f"Bootstrap Std: "f"{bootstrap_sharpes.std():.2f}")
    print(f"Train Sharpe: "f"{train_sharpe:.2f}")
    print(f"Test Sharpe: "f"{test_sharpe:.2f}")
    print(f"Parameter Stability Std: "f"{sensitivity_df['Sharpe'].std():.2f}")
    print(f"\nRobustness Score: "f"{score}/100")
    print("\nSensitivity Results")
    print(sensitivity_df)

    # Bootstrap Plot
    plt.figure(figsize=(8, 5))
    plt.hist(bootstrap_sharpes,bins=30)
    plt.axvline(actual_sharpe,linestyle="--",label=f"Actual Sharpe = {actual_sharpe:.2f}")
    plt.title("Bootstrap Sharpe Distribution")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()