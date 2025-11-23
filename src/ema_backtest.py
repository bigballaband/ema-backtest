import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# Config
# ------------------------
# These are the main settings you can change at the top of the file

START_DATE = "2015-01-01"   # start date for pulling market data
TICKER = "SPY"              # which symbol to test
FAST = 50                   # fast EMA window
SLOW = 200                  # slow EMA window
COST_PER_TRADE = 0.001      # trading cost per 1 unit of position change (0.1 percent)
ANNUAL_RF = 0.02            # annual risk free rate used in Sharpe
TRADING_DAYS = 252          # trading days per year for annualization


# ------------------------
# Data loading
# ------------------------
def get_price_data(ticker: str, start: str) -> pd.DataFrame:
    """
    Get price data from Yahoo Finance.

    We keep:
    - price: Adjusted Close (handles splits and dividends)
    - High, Low, Close: used later for ATR and other indicators
    """

    t = yf.Ticker(ticker)
    data = t.history(start=start, auto_adjust=False)

    # Make sure required columns exist
    # price is the main series we use for EMAs and returns
    if {"Adj Close", "High", "Low", "Close"}.issubset(data.columns):
        df = data[["Adj Close", "High", "Low", "Close"]].copy()
        df.rename(columns={"Adj Close": "price"}, inplace=True)
    else:
        # Fallback case, if for some reason Adj Close is missing
        df = data[["Close"]].copy()
        df.rename(columns={"Close": "price"}, inplace=True)

    return df


# ------------------------
# EMA signals
# ------------------------
def add_ema_signals(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    long_short: bool = True,
) -> pd.DataFrame:
    """
    Add EMA based trading signals.

    Inputs:
    - df: must contain 'price'
    - fast, slow: EMA windows
    - long_short:
        True  -> use 1 for long, -1 for short
        False -> use 1 for long, 0 otherwise (long only)

    Output:
    - df with columns:
        ema_fast, ema_slow, signal, position
    """

    df = df.copy()

    # Compute fast and slow EMAs on price
    df["ema_fast"] = df["price"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["price"].ewm(span=slow, adjust=False).mean()

    # Build signal series
    if long_short:
        # Long short version, sign can be -1, 0, or 1
        df["signal"] = 0
        df.loc[df["ema_fast"] > df["ema_slow"], "signal"] = 1
        df.loc[df["ema_fast"] < df["ema_slow"], "signal"] = -1
    else:
        # Long only version, only 0 or 1
        df["signal"] = (df["ema_fast"] > df["ema_slow"]).astype(int)

    # Shift by 1 day to avoid lookahead
    # We use today's signal to decide tomorrow's position
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


# ------------------------
# Backtest core
# ------------------------
def run_backtest(
    df: pd.DataFrame,
    cost_per_trade: float,
    annual_rf: float,
    trading_days: int = 252,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the backtest given a DataFrame with:
    - price
    - position (our exposure each day)

    Steps:
    - compute asset return
    - compute raw strategy return
    - apply trading costs when position changes
    - build equity curves
    - compute drawdowns
    - compute summary stats
    """

    df = df.copy()

    # Daily asset returns as percent changes
    df["asset_ret"] = df["price"].pct_change().fillna(0)

    # Raw strategy return before costs, just position * asset return
    df["strategy_ret_raw"] = df["position"] * df["asset_ret"]

    # How much the position changed day to day
    # 0 = no change, 1 = open or close, 2 = full flip from long to short
    df["trade"] = df["position"].diff().abs().fillna(0)

    # Trading cost scales with how much we changed the position
    df["cost"] = df["trade"] * cost_per_trade

    # Net return after subtracting cost
    df["strategy_ret"] = df["strategy_ret_raw"] - df["cost"]

    # Equity curves for strategy and buy and hold
    df["strategy_equity"] = (1 + df["strategy_ret"]).cumprod()
    df["asset_equity"] = (1 + df["asset_ret"]).cumprod()

    # Track running peak of strategy equity
    df["peak"] = df["strategy_equity"].cummax()

    # Drawdown is the percent drop from the peak
    df["drawdown"] = df["strategy_equity"] / df["peak"] - 1

    # Summary stats dictionary
    stats = compute_stats(df, annual_rf=annual_rf, trading_days=trading_days)

    return df, stats


# Extra stats about trading behavior
def extra_stats(df: pd.DataFrame) -> dict:
    """
    Some extra descriptive stats:
    - total position changes (trade units)
    - days in long
    - days in short
    - percent of time long or short
    """

    stats = {}

    # Sum of trade units, rough measure of how much we trade
    stats["n_trades_units"] = df["trade"].sum()

    # How many days we hold long or short positions
    stats["days_long"] = (df["position"] > 0).sum()
    stats["days_short"] = (df["position"] < 0).sum()

    # Fraction of time long or short
    stats["pct_time_long"] = (df["position"] > 0).mean()
    stats["pct_time_short"] = (df["position"] < 0).mean()

    return stats


# ------------------------
# Stats builder
# ------------------------
def compute_stats(
    df: pd.DataFrame,
    annual_rf: float,
    trading_days: int = 252,
) -> dict:
    """
    Compute summary stats for the strategy.

    Returns a dict with:
    - max_drawdown
    - total_return_strategy
    - total_return_buy_hold
    - sharpe
    """

    stats = {}

    # Max drawdown is the worst drop from peak equity
    stats["max_drawdown"] = df["drawdown"].min()

    # Total compounded return from start to end
    stats["total_return_strategy"] = df["strategy_equity"].iloc[-1] - 1
    stats["total_return_buy_hold"] = df["asset_equity"].iloc[-1] - 1

    # Daily risk free rate from annual rate
    daily_rf = (1 + annual_rf) ** (1 / trading_days) - 1

    # Extra return above risk free
    excess = df["strategy_ret"] - daily_rf

    # Sharpe ratio with annual scaling, guard against zero std
    if excess.std() > 0:
        sharpe = np.sqrt(trading_days) * excess.mean() / excess.std()
    else:
        sharpe = np.nan
    stats["sharpe"] = sharpe

    return stats


# ------------------------
# Plot helpers
# ------------------------
def plot_price(df: pd.DataFrame) -> None:
    """
    Plot price with fast and slow EMAs on top.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["price"], label="price")
    plt.plot(df.index, df["ema_fast"], label="fast EMA")
    plt.plot(df.index, df["ema_slow"], label="slow EMA")
    plt.legend()
    plt.title("Price with EMAs")
    plt.grid(True)
    plt.show()


def plot_signals(df: pd.DataFrame) -> None:
    """
    Plot trade entries and exits on top of price.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["price"], label="price", alpha=0.7)

    # points where we go from 0 to 1 (or -1)
    entries = df[df["position"].diff() == 1]
    # points where we go from 1 to 0 (or 0 to -1)
    exits = df[df["position"].diff() == -1]

    plt.scatter(entries.index, entries["price"], color="green", label="entry", s=30)
    plt.scatter(exits.index, exits["price"], color="red", label="exit", s=30)

    plt.legend()
    plt.title("Trade signals")
    plt.grid(True)
    plt.show()


def plot_equity(df: pd.DataFrame) -> None:
    """
    Plot equity curves for strategy and buy and hold.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["strategy_equity"], label="strategy")
    plt.plot(df.index, df["asset_equity"], label="buy and hold")
    plt.legend()
    plt.title("Equity curve")
    plt.grid(True)
    plt.show()


def plot_drawdown(df: pd.DataFrame) -> None:
    """
    Plot drawdown over time.
    """

    plt.figure(figsize=(12, 4))
    plt.plot(df.index, df["drawdown"], label="drawdown")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Drawdown")
    plt.grid(True)
    plt.show()


# ------------------------
# Parameter sweep
# ------------------------
def sweep_ema_params(
    df_raw: pd.DataFrame,
    fast_list: list[int],
    slow_list: list[int],
    long_short: bool,
    cost_per_trade: float,
    annual_rf: float,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Run the EMA backtest on many fast/slow pairs.
    Lets you see which combinations give better stats.

    Returns a DataFrame where each row is one EMA pair with its stats.
    """

    rows = []

    # Loop through every fast and slow option
    for fast in fast_list:
        for slow in slow_list:

            # Fast EMA must be smaller than slow EMA or crossover logic is weird
            if fast >= slow:
                continue

            # Build signals for this pair
            df = add_ema_signals(df_raw, fast=fast, slow=slow, long_short=long_short)

            # Run backtest
            df_bt, stats = run_backtest(
                df,
                cost_per_trade=cost_per_trade,
                annual_rf=annual_rf,
                trading_days=trading_days,
            )

            # Save the results
            rows.append({
                "fast": fast,
                "slow": slow,
                "long_short": long_short,
                **stats,
            })

    # Return results as a clean table
    return pd.DataFrame(rows)


# ------------------------
# ATR and volatility based features
# ------------------------
def add_atr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """
    Add ATR (Average True Range) to the data.

    ATR is the average of the True Range over the last n days.
    True Range looks at:
    - daily high minus low
    - gap up from yesterday's close
    - gap down from yesterday's close
    """

    df = df.copy()

    # Need High, Low, Close for ATR
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Yesterday's close
    prev_close = close.shift(1)

    # Three types of true range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # True range is the max of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is moving average of true range
    df["ATR"] = tr.rolling(n).mean()

    return df


def add_position_sizing(
    df: pd.DataFrame,
    risk_target: float = 0.01,
    max_leverage: float = 1.0
) -> pd.DataFrame:
    """
    Adjust position size based on ATR.

    Idea:
    - when ATR is small, we can size up
    - when ATR is large, we size down
    This aims for more steady risk per day.
    """

    df = df.copy()

    # Avoid divide by zero
    atr = df["ATR"].replace(0, np.nan)

    # Raw size is target risk divided by ATR
    raw_size = risk_target / atr

    # Clip to max_leverage so we do not go above a limit
    size = raw_size.clip(upper=max_leverage).fillna(0)

    # Multiply direction by size
    # direction comes from signal, size from ATR
    df["position"] = df["signal"].shift(1).fillna(0) * size

    return df


# ------------------------
# Walk forward testing
# ------------------------
def walk_forward_once(
    df_raw: pd.DataFrame,
    train_end: str,
    fast_list: list[int],
    slow_list: list[int],
    long_short: bool,
    cost_per_trade: float,
    annual_rf: float,
):
    """
    Single walk forward test.

    Steps:
    1. Use data up to train_end to pick best EMA pair.
    2. Use that pair on data after train_end.
    3. Return test equity, test stats, and training sweep results.
    """

    # Split into training and test sets
    df_train = df_raw.loc[:train_end]
    df_test = df_raw.loc[train_end:]

    # Run parameter sweep on the training set
    train_results = sweep_ema_params(
        df_train,
        fast_list=fast_list,
        slow_list=slow_list,
        long_short=long_short,
        cost_per_trade=cost_per_trade,
        annual_rf=annual_rf,
    )

    # Pick highest Sharpe on training
    best = train_results.sort_values("sharpe", ascending=False).iloc[0]
    best_fast = int(best["fast"])
    best_slow = int(best["slow"])

    # Build signals on test data with the best pair
    df_test_sig = add_ema_signals(df_test, fast=best_fast, slow=best_slow, long_short=long_short)

    # Run test backtest
    df_test_bt, test_stats = run_backtest(
        df_test_sig,
        cost_per_trade=cost_per_trade,
        annual_rf=annual_rf,
    )

    return df_test_bt, test_stats, train_results


# ------------------------
# Multi asset support
# ------------------------
def backtest_on_universe(
    tickers: list[str],
    start: str,
    fast: int,
    slow: int,
    long_short: bool,
    cost_per_trade: float,
    annual_rf: float,
):
    """
    Run the same EMA strategy on many tickers.

    Returns a dict: {ticker: backtest_df}
    """

    dfs = {}

    for ticker in tickers:
        # Load data
        df_raw = get_price_data(ticker, start)

        # Build signals
        df_sig = add_ema_signals(df_raw, fast=fast, slow=slow, long_short=long_short)

        # Backtest
        df_bt, stats = run_backtest(
            df_sig,
            cost_per_trade=cost_per_trade,
            annual_rf=annual_rf,
        )

        dfs[ticker] = df_bt

    return dfs


# ------------------------
# Triple EMA signals
# ------------------------
def add_triple_ema_signals(
    df: pd.DataFrame,
    fast: int,
    mid: int,
    slow: int,
    long_short: bool = False,
) -> pd.DataFrame:
    """
    Triple EMA strategy.

    Only trade when all three EMAs are in clear order.

    - Uptrend: ema_fast > ema_mid > ema_slow
    - Downtrend (if long_short is True): ema_fast < ema_mid < ema_slow
    """

    df = df.copy()

    # Three EMAs of different speeds
    df["ema_fast"] = df["price"].ewm(span=fast, adjust=False).mean()
    df["ema_mid"]  = df["price"].ewm(span=mid, adjust=False).mean()
    df["ema_slow"] = df["price"].ewm(span=slow, adjust=False).mean()

    df["signal"] = 0

    # Uptrend case
    df.loc[
        (df["ema_fast"] > df["ema_mid"]) &
        (df["ema_mid"]  > df["ema_slow"]),
        "signal"
    ] = 1

    if long_short:
        # Downtrend case
        df.loc[
            (df["ema_fast"] < df["ema_mid"]) &
            (df["ema_mid"]  < df["ema_slow"]),
            "signal"
        ] = -1

    # Shift to avoid lookahead
    df["position"] = df["signal"].shift(1).fillna(0)

    return df


# ------------------------
# Main
# ------------------------
def main():
    # 1. Load raw data
    df_raw = get_price_data(TICKER, START_DATE)

    # 2. Add EMA signals
    # Set long_short=False if you want long only behavior
    df = add_ema_signals(df_raw, fast=FAST, slow=SLOW, long_short=True)

    # 3. Run backtest on this setup
    df_bt, stats = run_backtest(
        df,
        cost_per_trade=COST_PER_TRADE,
        annual_rf=ANNUAL_RF,
        trading_days=TRADING_DAYS,
    )

    # 4. Extra stats about trading behavior
    more = extra_stats(df_bt)
    print("\nExtra stats:")
    for k, v in more.items():
        print(f"{k:25s}: {v}")

    # 5. Summary backtest stats
    print("\nBacktest stats:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v: .4f}")
        else:
            print(f"{k:25s}: {v}")

    # 6. Plots
    plot_price(df_bt)
    plot_signals(df_bt)
    plot_equity(df_bt)
    plot_drawdown(df_bt)


if __name__ == "__main__":
    main()
