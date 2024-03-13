import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ProcessPoolExecutor

def binomial_tree_option_price(S, K, T, r, sigma, N=50_000, dividends=0, skew=0, kurtosis=0):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt) + skew * sigma * dt)  # Adjusted for skew
    d = np.exp(-sigma * np.sqrt(dt) + skew * sigma * dt)  # Adjusted for skew
    p = (np.exp((r - dividends) * dt) - d) / (u - d)  # Adjusted for dividends
    
    # Adjust p slightly to account for kurtosis (heuristic adjustment)
    p += kurtosis * 0.001 * (1 - 2 * p)  # This is a heuristic adjustment

    prices = S * d**np.arange(N, -1, -1) * u**np.arange(0, N+1, 1)
    option_values = np.maximum(prices - K, 0)
    
    discount_factor = np.exp(-r * dt)
    for i in range(N - 1, -1, -1):
        option_values[:i+1] = (p * option_values[1:i+2] + (1 - p) * option_values[:i+1]) * discount_factor
        option_values[:i+1] = np.maximum(option_values[:i+1], prices[:i+1] - K)
    
    return option_values[0]

def calculate_option_price(params):
    S, K, T, r, sigma = params
    return binomial_tree_option_price(S, K, T, r, sigma)

if __name__ == '__main__':
    # Retrieve historical data
    ticker = "BTCC-B.TO"
    start_time = (datetime.now(pytz.timezone('US/Pacific')) - timedelta(days=365*4)).strftime('%Y-%m-%d')
    end_time = datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_time, end=end_time, interval="1d")[['Close']]
    data = data.copy()
    data.loc[:, 'Daily_Return'] = data['Close'].pct_change()
    daily_volatility = data['Daily_Return'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)

    # Risk-free rate (annualized)
    risk_free_rate = 0.05

    # Current stock price
    current_stock_price = data['Close'].iloc[-1]

    # Calculate the next 4 Fridays from today, spaced two weeks apart
    today = datetime.now(pytz.timezone('US/Pacific')).date()
    friday_dates = [today + timedelta(days=((4 - today.weekday()) % 7) + 14 * i) for i in range(4)]

    # Generate a range of strike prices
    strike_price_range = np.arange(0, 0.55, 0.05)
    strike_prices = [current_stock_price * (1 + x) for x in strike_price_range]

    # Prepare parameters for parallel execution
    tasks = [(current_stock_price, strike_price, (expiry_date - today).days / 365, risk_free_rate, annualized_volatility) for expiry_date in friday_dates for strike_price in strike_prices]
    
    # Use ProcessPoolExecutor to parallelize option price calculation
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calculate_option_price, task) for task in tasks]
        for i, future in enumerate(futures):
            expiry_date, strike_price = tasks[i][0:2]
            option_price = future.result()
            results.append({'Expiration Date': expiry_date, 'Strike Price': strike_price, 'Call Option Price': option_price})

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    csv_filename = 'option_prices.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")
