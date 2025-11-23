import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# Config
# ------------------------

START_DATE = "2015-01-01"
TICKER = "SPY"
FAST = 50
SLOW = 200
COST_PER_TRADE = 0.001
ANNUAL_RF = 0.02
TRADING_DAYS = 252


def get_price_data(ticker: str, start: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    data = t.history(start=start, auto_adjust=False)

    if "Adj Close" in data.columns:
        df = data[["Adj Close"]].copy()
        df.rename(columns={"Adj Close": "price"}, inplace=True)
    else:
        df = data[["Close"]].copy()
        df.rename(columns={"Close": "price"}, inplace=True)

    return df


def add_ema_signals(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    long_short: bool = True,
) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast"] = df["price"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["price"].ewm(span=slow, adjust=False).mean()

    if long_short:
        df["signal"] = 0
        df.loc[df["ema_fast"] > df["ema_slow"], "signal"] = 1
        df.loc[df["ema_fast"] < df["ema_slow"], "signal"] = -1
    else:
        df["signal"] = (df["ema_fast"] > df["ema_slow"]).astype(int)

    # shift to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


def run_backtest(
    df: pd.DataFrame,
    cost_per_trade: float,
    annual_rf: float,
    trading_days: int = 252,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    # returns
    df["asset_ret"] = df["price"].pct_change().fillna(0)
    df["strategy_ret_raw"] = df["position"] * df["asset_ret"]

    # trades and costs
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["cost"] = df["trade"] * cost_per_trade
    df["strategy_ret"] = df["strategy_ret_raw"] - df["cost"]

    # equity curves
    df["strategy_equity"] = (1 + df["strategy_ret"]).cumprod()
    df["asset_equity"] = (1 + df["asset_ret"]).cumprod()

    # drawdown
    df["peak"] = df["strategy_equity"].cummax()
    df["drawdown"] = df["strategy_equity"] / df["peak"] - 1

    # stats
    stats = compute_stats(df, annual_rf=annual_rf, trading_days=trading_days)

    return df, stats

def extra_stats(df):
    stats = {}

    stats["n_trades_units"] = df["trade"].sum()
    stats["days_long"] = (df["position"] > 0).sum()
    stats["days_short"] = (df["position"] < 0).sum()
    stats["pct_time_long"] = (df["position"] > 0).mean()
    stats["pct_time_short"] = (df["position"] < 0).mean()

    return stats


def compute_stats(
    df: pd.DataFrame,
    annual_rf: float,
    trading_days: int = 252,
) -> dict:
    stats = {}

    # max drawdown
    stats["max_drawdown"] = df["drawdown"].min()

    # total returns
    stats["total_return_strategy"] = df["strategy_equity"].iloc[-1] - 1
    stats["total_return_buy_hold"] = df["asset_equity"].iloc[-1] - 1

    # Sharpe
    daily_rf = (1 + annual_rf) ** (1 / trading_days) - 1
    excess = df["strategy_ret"] - daily_rf
    if excess.std() > 0:
        sharpe = np.sqrt(trading_days) * excess.mean() / excess.std()
    else:
        sharpe = np.nan
    stats["sharpe"] = sharpe

    return stats



def plot_price(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["price"], label="price")
    plt.plot(df.index, df["ema_fast"], label="fast EMA")
    plt.plot(df.index, df["ema_slow"], label="slow EMA")
    plt.legend()
    plt.title("Price with EMAs")
    plt.grid(True)
    plt.show()


def plot_signals(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["price"], label="price", alpha=0.7)

    entries = df[df["position"].diff() == 1]
    exits = df[df["position"].diff() == -1]

    plt.scatter(entries.index, entries["price"], color="green", label="entry", s=30)
    plt.scatter(exits.index, exits["price"], color="red", label="exit", s=30)

    plt.legend()
    plt.title("Trade signals")
    plt.grid(True)
    plt.show()


def plot_equity(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["strategy_equity"], label="strategy")
    plt.plot(df.index, df["asset_equity"], label="buy and hold")
    plt.legend()
    plt.title("Equity curve")
    plt.grid(True)
    plt.show()


def plot_drawdown(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["drawdown"], label="drawdown")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Drawdown")
    plt.grid(True)
    plt.show()


def main():
    # 1. Load data
    df = get_price_data(TICKER, START_DATE)

    # 2. Build signals (set long_short to False if you want long only)
    df = add_ema_signals(df, fast=FAST, slow=SLOW, long_short=True)

    # 3. Run backtest
    df, stats = run_backtest(
        df,
        cost_per_trade=COST_PER_TRADE,
        annual_rf=ANNUAL_RF,
        trading_days=TRADING_DAYS,
    )
    
    more = extra_stats(df)
    print("\nExtra stats:")
    for k, v in more.items():
        print(f"{k:25s}: {v}")


    # 4. Print stats
    print("Backtest stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v: .4f}")
        else:
            print(f"{k:25s}: {v}")

    # 5. Plots
    plot_price(df)
    plot_signals(df)
    plot_equity(df)
    plot_drawdown(df)


if __name__ == "__main__":
    main()

