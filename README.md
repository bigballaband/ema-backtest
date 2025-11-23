# EMA Crossover Backtest

A simple EMA crossover backtest in Python on SPY that:

- downloads price data with `yfinance`
- builds fast and slow EMAs
- creates long only or long/short signals
- applies trading costs
- computes equity curve, max drawdown, and Sharpe ratio
- plots price with EMAs, signals, equity curve, and drawdown

## Setup

```bash
git clone https://github.com/bigballaband/ema-backtest
cd ema-backtest
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
