import pandas as pd
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def rolling_stoch_rsi(
    df, column="price", period=60, k_period=6, d_period=6, ma_period=60
):
    # Initialize columns for RSI, Stochastic RSI, and Moving Average
    df["rsi"] = np.nan
    df["stoch_rsi_k"] = np.nan
    df["stoch_rsi_d"] = np.nan
    df["mov_avg"] = (
        df[column].rolling(window=ma_period).mean()
    )  # 120-period moving average

    # Track buy/sell positions based on MA trend
    df["position"] = None  # 'buy', 'sell', or None

    for i in range(period, len(df)):
        current_frame = df.iloc[i - period + 1 : i + 1]
        delta = current_frame[column].diff(1)
        gain = delta.where(delta > 0, 0).mean()
        loss = -delta.where(delta < 0, 0).mean()

        rs = gain / loss if loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        df.iloc[i, df.columns.get_loc("rsi")] = rsi

        if i >= period + k_period - 1:
            rsi_frame = df["rsi"].iloc[i - k_period + 1 : i + 1]
            min_rsi = rsi_frame.min()
            max_rsi = rsi_frame.max()
            stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-10) * 100
            df.iloc[i, df.columns.get_loc("stoch_rsi_k")] = stoch_rsi

            if i >= period + k_period + d_period - 2:
                k_frame = df["stoch_rsi_k"].iloc[i - d_period + 1 : i + 1]
                df.iloc[i, df.columns.get_loc("stoch_rsi_d")] = k_frame.mean()

    # Determine buy/sell positions based on MA trend
    for i in range(ma_period, len(df)):
        ma_diff = df.iloc[i - ma_period + 1 : i + 1]["mov_avg"].diff()
        if all(ma_diff > 0):
            df.iloc[i, df.columns.get_loc("position")] = "buy"
        elif all(ma_diff < 0):
            df.iloc[i, df.columns.get_loc("position")] = "sell"

    # Implement logic to hold until MA peaks for buys and bottoms for sells
    df["action"] = None  # Placeholder for buy/sell actions
    holding = False
    for i in range(len(df)):
        if holding and df.iloc[i]["position"] == "sell":
            df.iloc[i, df.columns.get_loc("action")] = "sell"
            holding = False
        elif not holding and df.iloc[i]["position"] == "buy":
            df.iloc[i, df.columns.get_loc("action")] = "buy"
            holding = True

        if holding:
            # If holding, check if the MA is peaking
            if i > 0 and df.iloc[i]["mov_avg"] < df.iloc[i - 1]["mov_avg"]:
                df.iloc[i, df.columns.get_loc("action")] = "sell"
                holding = False


# Execute Buy and Sell Strategies
def execute_buy_sell_strategies(
    df,
    market_phase,
    initial_funds=10_000,
    funds_allocation=0.2,
    price_increase_threshold=1.03,
    stoch_rsi_threshold_low=30,
    stoch_rsi_threshold_high=70,
    commission_percent=1.2,
):
    current_funds = initial_funds
    total_profit, buy_signals, capital_deployed, profits = 0, [], [], []
    trade_count = 0  # Initialize trade counter

    for date, row in df.iterrows():
        stoch_rsi_k = row["stoch_rsi_k"]
        price = row["price"]
        available_funds = current_funds + sum(
            profits
        )  # Update available funds to include profits

        if stoch_rsi_k < stoch_rsi_threshold_low and available_funds > 0:
            funds_to_use = available_funds * funds_allocation
            shares_bought = funds_to_use / price
            current_funds -= funds_to_use
            buy_signals.append((date, price, shares_bought, funds_to_use))
            capital_deployed.append(funds_to_use)  # Track capital deployed for each buy
            trade_count += 1
        elif stoch_rsi_k > stoch_rsi_threshold_high and buy_signals:
            updated_buy_signals = []
            for buy_signal in buy_signals:
                buy_date, buy_price, shares, funds_used = buy_signal
                if price >= buy_price * price_increase_threshold:
                    gross_sell_amount = shares * price
                    commission = gross_sell_amount * (commission_percent / 100)
                    net_sell_amount = gross_sell_amount - commission
                    profit = net_sell_amount - funds_used
                    total_profit += profit
                    current_funds += funds_used
                    profits.append(profit)
                    trade_count += 1  # Count sell action as a trade
                else:
                    updated_buy_signals.append(buy_signal)
            buy_signals = updated_buy_signals

    # Calculate average capital deployed
    average_capital_deployed = np.mean(capital_deployed) if capital_deployed else 0

    return total_profit, current_funds, trade_count, average_capital_deployed


def backtest_trading_strategy(df, market_phase, initial_funds=10_000):
    total_profit, final_funds, trade_count, average_capital_deployed = (
        execute_buy_sell_strategies(df, market_phase, initial_funds=initial_funds)
    )
    buy_and_hold_profit = calculate_buy_and_hold_profit(df, initial_funds=initial_funds)

    print(f"[{market_phase}] Trading Strategy Total Profit: {total_profit}")
    print(f"[{market_phase}] Buy and Hold Strategy Profit: {buy_and_hold_profit}")
    print(f"[{market_phase}] Number of Trades Executed: {trade_count}")
    print(f"[{market_phase}] Average Capital Deployed: {average_capital_deployed}")

    trading_days = (df.index.max() - df.index.min()).days + 1
    average_profit_per_day = total_profit / trading_days
    print(
        f"[{market_phase}] Trading Strategy Average Profit per Day: {average_profit_per_day}"
    )

    if total_profit > buy_and_hold_profit:
        print(
            f"[{market_phase}] The trading strategy outperformed the buy-and-hold strategy."
        )
    elif total_profit < buy_and_hold_profit:
        print(
            f"[{market_phase}] The buy-and-hold strategy outperformed the trading strategy."
        )
    else:
        print(f"[{market_phase}] Both strategies resulted in the same total profit.")


def load_and_prepare_data():
    df = pd.read_csv("btc-data.csv")
    df["Date"] = (
        pd.to_datetime(df["Timestamp"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert("America/Vancouver")
    )
    df.set_index("Date", inplace=True)
    df["price"] = df["Weighted_Price"].interpolate(
        method="linear", limit_direction="both"
    )
    df = df[["price"]]
    return df


def prepare_and_analyze_market_phase(phase, df):
    # Assuming df_phase is prepared based on the phase
    if phase == "bull":
        df_phase = df.loc["2020-09-01":"2021-01-01"]
    elif phase == "bear":
        df_phase = df.loc["2018-09-01":"2019-01-01"]
    else:  # chop
        df_phase = df.loc["2019-09-01":"2020-01-01"]

    rolling_stoch_rsi(df_phase)
    # Now also pass the market phase to backtest_trading_strategy
    backtest_trading_strategy(df_phase, phase)


def calculate_buy_and_hold_profit(df, initial_funds=10_000):
    # Calculate the total number of shares that could be bought at the start
    initial_price = df["price"].iloc[0]
    shares_bought = initial_funds / initial_price

    # Calculate the total value at the end of the period
    final_price = df["price"].iloc[-1]
    final_value = shares_bought * final_price

    # Calculate profit
    profit = final_value - initial_funds
    return profit


def main():
    start = time.time()

    df = load_and_prepare_data()

    # Ensure the ProcessPoolExecutor correctly utilizes multiple cores for each market phase analysis
    with ProcessPoolExecutor(max_workers=9) as executor:
        futures = [
            executor.submit(prepare_and_analyze_market_phase, phase, df)
            for phase in ["bull", "bear", "chop"]
        ]
        # Wait for all submitted tasks to complete
        for future in futures:
            future.result()

    end = time.time()
    print("\nExecution time: ", str(end - start), "seconds")


if __name__ == "__main__":
    main()
