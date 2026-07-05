# Signal Robustness & Overfitting Diagnostics

**Author:** Dzandu Selorm(dzand-cmd)  
**Project Type:** Quantitative Research  
**Language:** Python  
**Status:** Complete  

---

## Overview

Trading signals often appear profitable in-sample but fail out-of-sample due to overfitting.
This project builds a systematic framework to answer:
Is a trading signal genuinely predictive or just noise-fitting?

It evaluates signal robustness using:
- Bootstrap resampling of strategy returns
- Train/test split validation
- Parameter sensitivity analysis
- Sharpe ratio stability assessment

---

## Project Structure

signal-robustness/
│
├──  signal_robustness.py
├── .gitignore
├── README.md


## Features

1. Signal Generation
- Moving average crossover strategy (configurable short/long windows)
  
2. Backtesting Engine
- Computes strategy returns from signal positions
- Incorporates realistic lag via shifted signals
  
3. Statistical Validation
- Bootstrap distribution of Sharpe ratios
- Confidence estimation for strategy performance
- Overfitting detection via resampling
  
4. Robustness Testing
- Train vs test performance comparison
- Sensitivity analysis across parameter combinations
- Stability measurement across regimes
  
5. Performance Metrics
- Sharpe Ratio (annualized)
- Mean & variance of bootstrap Sharpe distribution
- Parameter stability statistics


## Methodology

1. Signal Construction
A simple moving average crossover signal:
- Buy when short MA > long MA
- Sell otherwise
  
2. Strategy Returns
- R(t) = S(t−1) * r(t)
where:
S(t): trading signal
r(t): asset returns

3. Bootstrap Testing
Random resampling of strategy returns to estimate:
- Expected Sharpe under noise
- Distribution of performance
- Statistical significance of observed Sharpe

4. Parameter Sensitivity
Tests multiple (short, long) MA combinations to evaluate:
- Stability of signal performance
- Dependence on hyperparameters
  
5. Train/Test Validation
Splits data into:
- In-sample (training)
- Out-of-sample (testing)
Used to detect overfitting.


## Key Insight

A profitable signal is not enough — it must be:
- stable across time
- robust across parameters
- statistically significant vs noise


## How To Run

- git clone https://github.com/your-username/signal-robustness.git
- cd signal-robustness
- pip install numpy pandas matplotlib yfinance
- python signal_robustness.py


