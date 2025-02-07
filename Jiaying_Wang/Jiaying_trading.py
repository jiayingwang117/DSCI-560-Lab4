import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import warnings
warnings.filterwarnings("ignore")  # to ignore ARIMA warnings for demonstration

# --- Database configuration ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # your MySQL username
    'password': 'Ashley020501',
    'database': 'stock_analysis'
}

# Create the SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

# --- Function to fetch stock data for a given portfolio ---
def fetch_portfolio_stocks(portfolio_id):
    """
    Retrieves stock data (symbol, date, close_price) for a given portfolio ID.
    """
    query = """
        SELECT ps.symbol, sd.date, sd.close_price 
        FROM portfolio_stocks ps
        JOIN stock_data sd ON ps.symbol = sd.symbol
        WHERE ps.portfolio_id = %s
        ORDER BY sd.date ASC
    """
    df = pd.read_sql(query, engine, params=(portfolio_id,))
    df['date'] = pd.to_datetime(df['date'])
    return df

# --- Function to apply the hybrid strategy on a single stock dataframe ---
def apply_hybrid_strategy(stock_df, short_window=20, long_window=50):
    """
    Applies a hybrid trading strategy to a stock's time series.
    The strategy combines:
      1. A simple Moving Average Crossover signal.
      2. An ARIMA-based forecast signal using out-of-sample predictions.
      
    The final (hybrid) signal is:
      - 1 (Buy) if both the moving average and ARIMA signals indicate an upward move.
      - -1 (Sell) if both indicate a downward move.
      - 0 (Hold) otherwise.
    """
    # Ensure the dataframe is sorted by date
    stock_df = stock_df.sort_index()

    # --- 1. Moving Average Signals ---
    stock_df['Short_MA'] = stock_df['close_price'].rolling(window=short_window, min_periods=1).mean()
    stock_df['Long_MA'] = stock_df['close_price'].rolling(window=long_window, min_periods=1).mean()
    
    # Signal: 1 if short-term MA is above long-term MA, else -1 if below.
    stock_df['Signal_MA'] = 0
    stock_df.loc[stock_df['Short_MA'] > stock_df['Long_MA'], 'Signal_MA'] = 1
    stock_df.loc[stock_df['Short_MA'] < stock_df['Long_MA'], 'Signal_MA'] = -1

    # --- 2. ARIMA Out-of-Sample Forecast and Signal ---
    # We will use a walk-forward approach: for each day (after an initial training period),
    # fit the ARIMA model using data up to that day and forecast the next day.
    min_train_size = long_window  # Ensure enough data; adjust as needed.
    predictions = [np.nan] * len(stock_df)
    
    for i in range(min_train_size, len(stock_df)-1):
        train_data = stock_df['close_price'].iloc[:i+1]
        try:
            model = ARIMA(train_data, order=(5,1,0))
            arima_fit = model.fit()
            forecast = arima_fit.forecast(steps=1)
            predictions[i+1] = forecast[0]
        except Exception as e:
            print(f"Error forecasting ARIMA at index {i+1} for symbol {stock_df['symbol'].iloc[0]}: {e}")
            predictions[i+1] = np.nan
            
    stock_df['ARIMA_Prediction'] = predictions

    # Calculate ARIMA error metrics on days with available predictions
    valid_idx = stock_df['ARIMA_Prediction'].notna()
    if valid_idx.sum() > 0:
        arima_mae = mean_absolute_error(stock_df.loc[valid_idx, 'close_price'], stock_df.loc[valid_idx, 'ARIMA_Prediction'])
        arima_rmse = math.sqrt(mean_squared_error(stock_df.loc[valid_idx, 'close_price'], stock_df.loc[valid_idx, 'ARIMA_Prediction']))
        print(f"ARIMA Evaluation -> MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}")
    else:
        print("No valid ARIMA predictions to evaluate.")

    # Generate ARIMA signal based on forecast trend:
    # Signal is 1 if the current out-of-sample prediction is higher than the previous prediction, else -1.
    stock_df['Signal_ARIMA'] = 0
    stock_df.loc[stock_df['ARIMA_Prediction'].shift(1) < stock_df['ARIMA_Prediction'], 'Signal_ARIMA'] = 1
    stock_df.loc[stock_df['ARIMA_Prediction'].shift(1) >= stock_df['ARIMA_Prediction'], 'Signal_ARIMA'] = -1

    # --- 3. Hybrid Signal ---
    # Only take a trade if both the MA and ARIMA methods agree.
    stock_df['Hybrid_Signal'] = 0
    condition_buy = (stock_df['Signal_MA'] == 1) & (stock_df['Signal_ARIMA'] == 1)
    condition_sell = (stock_df['Signal_MA'] == -1) & (stock_df['Signal_ARIMA'] == -1)
    stock_df.loc[condition_buy, 'Hybrid_Signal'] = 1
    stock_df.loc[condition_sell, 'Hybrid_Signal'] = -1

    return stock_df

# --- Function to simulate trades given a stock dataframe with signals ---
def simulate_trades(stock_df, initial_capital=10000, stop_loss_pct=0.05, take_profit_pct=0.10):
    """
    Simulates trading over time using an "all-in/all-out" strategy with risk management:
      - When the signal is 1 (buy) and cash is available, use all cash to buy shares.
      - When a position is held, if the price drops below the stop-loss threshold (e.g. 5% below entry)
        or rises above the take-profit threshold (e.g. 10% above entry), exit the trade immediately.
      - Also exit when the signal is -1 (sell) and shares are held.
      - When the signal is 0, hold the current position.
      
    The function updates the dataframe with a 'Portfolio_Value' column.
    """
    cash = initial_capital
    shares = 0
    buy_price = None  # Record the price at which shares were purchased
    portfolio_values = []
    
    # Loop through each time step
    for i in range(len(stock_df)):
        price = stock_df['close_price'].iloc[i]
        signal = stock_df['Hybrid_Signal'].iloc[i]
        
        # If currently holding shares, check for risk management exits.
        if shares > 0:
            # Check for stop loss: if current price is below (entry price * (1 - stop_loss_pct))
            if price <= buy_price * (1 - stop_loss_pct):
                cash = shares * price
                shares = 0
                buy_price = None
            # Check for take profit: if current price is above (entry price * (1 + take_profit_pct))
            elif price >= buy_price * (1 + take_profit_pct):
                cash = shares * price
                shares = 0
                buy_price = None
            # Also, if a sell signal is generated, exit the position.
            elif signal == -1:
                cash = shares * price
                shares = 0
                buy_price = None
        
        # If not holding any shares and a buy signal is received, enter a position.
        else:
            if signal == 1:
                shares = cash / price
                cash = 0
                buy_price = price
                # print(f"Buy signal executed on {stock_df.index[i].date()} at price {price:.2f}")
        
        # Update the current portfolio value.
        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)
    
    stock_df['Portfolio_Value'] = portfolio_values
    return stock_df

# --- Function to calculate overall performance metrics ---
def calculate_performance(portfolio_dfs, initial_capital=10000):
    """
    Aggregates the individual stock portfolio values by date, then calculates:
      - Final Portfolio Value
      - Annualized Return (assuming 252 trading days)
      - Sharpe Ratio (assuming a risk-free rate of 0)
    """
    # Combine portfolio values from all stocks by date.
    combined_df = pd.concat([df[['Portfolio_Value']] for df in portfolio_dfs], axis=1)
    combined_df['Total_Portfolio_Value'] = combined_df.sum(axis=1)
    combined_df.sort_index(inplace=True)
    
    # Performance metrics
    final_value = combined_df['Total_Portfolio_Value'].iloc[-1]
    n_days = combined_df.index.nunique()
    n_stocks = len(portfolio_dfs)
    total_initial_capital = initial_capital * n_stocks
    
    annualized_return = (final_value / total_initial_capital) ** (252 / n_days) - 1
    daily_returns = combined_df['Total_Portfolio_Value'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else np.nan

    print("\n=== Overall Performance Metrics ===")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return combined_df

# --- Main Execution ---
def main():
    # 1. Get portfolio id input from user
    try:
        portfolio_id = int(input("Enter Portfolio ID: "))
    except ValueError:
        print("Invalid portfolio ID. Please enter an integer value.")
        return
    
    # 2. Fetch the portfolio data
    df = fetch_portfolio_stocks(portfolio_id)
    if df.empty:
        print("No data found for the given portfolio ID.")
        return
    
    # 3. Process each symbol separately
    portfolio_results = []  # to store processed dataframes for each stock
    symbols = df['symbol'].unique()
    for symbol in symbols:
        # Filter data for the symbol and set date as index
        stock_df = df[df['symbol'] == symbol].copy()
        stock_df.set_index('date', inplace=True)
        stock_df.sort_index(inplace=True)
        stock_df['symbol'] = symbol  # add symbol column for clarity
        
        # Resample to business days and interpolate missing close prices
        stock_df = stock_df.asfreq('B')
        stock_df['close_price'] = stock_df['close_price'].interpolate()
        
        print(f"\nProcessing symbol: {symbol}")
        # 4. Apply the hybrid strategy to generate signals (including out-of-sample ARIMA predictions)
        stock_df = apply_hybrid_strategy(stock_df, short_window=20, long_window=50)
        
        # 5. Simulate trades with risk management rules (adjust stop_loss_pct and take_profit_pct as needed)
        stock_df = simulate_trades(stock_df, initial_capital=10000, stop_loss_pct=0.05, take_profit_pct=0.10)
        
        # Append the resulting dataframe for later performance aggregation
        portfolio_results.append(stock_df)
    
    if not portfolio_results:
        print("No stocks were processed. Exiting.")
        return

    # 6. Calculate and display overall performance metrics
    overall_df = calculate_performance(portfolio_results, initial_capital=10000)
    
    # 7. Plot the overall portfolio value over time
    plt.figure(figsize=(12,6))
    plt.plot(overall_df.index, overall_df['Total_Portfolio_Value'], label='Overall Portfolio Value', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title("Overall Portfolio Value Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
