import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate Trading Based on LSTM Predictions
def simulate_trading(df, y_pred, initial_balance=10000, stop_loss=0.90, take_profit=1.15, position_size=0.15, min_hold_days=3):
    df = df.iloc[len(df) - len(y_pred):].copy()
    df['Prediction'] = y_pred  # LSTM predicted labels
    balance = initial_balance
    shares = 0
    buy_price = 0
    portfolio_value = []
    daily_returns = []
    hold_days = 0  # Track how long we hold a stock

    for index, row in df.iterrows():
        close_price = row['close_price']
        position_amount = balance * position_size  # Only risk 15% per trade

        # HOLD for a minimum period before selling
        if shares > 0:
            hold_days += 1

        # BUY Condition (Only if enough cash & good signal)
        if row['Prediction'] == 1 and balance >= position_amount and shares == 0:
            shares = position_amount / close_price
            balance -= position_amount
            buy_price = close_price
            hold_days = 0  # Reset holding period

        # SELL Condition (If prediction = 0 & held for min days)
        elif row['Prediction'] == 0 and shares > 0 and hold_days >= min_hold_days:
            balance += shares * close_price
            shares = 0

        # Stop-Loss Condition (Only sell if held for at least min days)
        elif shares > 0 and close_price < buy_price * stop_loss and hold_days >= min_hold_days:
            balance += shares * close_price
            shares = 0

        # Take-Profit Condition
        elif shares > 0 and close_price > buy_price * take_profit and hold_days >= min_hold_days:
            balance += shares * close_price
            shares = 0

        # Track portfolio value
        portfolio_val = balance + (shares * close_price)
        if portfolio_value:
            daily_returns.append((portfolio_val - portfolio_value[-1]) / portfolio_value[-1])
        portfolio_value.append(portfolio_val)

    df['Portfolio_Value'] = portfolio_value
    df['Daily_Returns'] = daily_returns + [0]  # Last day has no return

    return df

# Calculate Trading Performance Metrics
def calculate_metrics(df):
    returns = df['Daily_Returns'].dropna()

    # Sharpe Ratio
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0

    # Cumulative Return
    cumulative_return = (df['Portfolio_Value'].iloc[-1] - df['Portfolio_Value'].iloc[0]) / df['Portfolio_Value'].iloc[0]

    # Max Drawdown (Worst peak-to-trough loss)
    peak = df['Portfolio_Value'].cummax()
    drawdown = (df['Portfolio_Value'] - peak) / peak
    max_drawdown = drawdown.min()

    # Win Rate (Percentage of profitable trades)
    win_rate = np.sum(df['Daily_Returns'] > 0) / len(df['Daily_Returns'])

    # Print Trading Performance Metrics
    print(f"\n🔍 Trading Strategy Performance Metrics (Updated):")
    print(f"✅ Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"✅ Cumulative Return: {cumulative_return:.2%}")
    print(f"✅ Max Drawdown: {max_drawdown:.2%}")
    print(f"✅ Win Rate: {win_rate:.2%}\n")

    return {
        "Sharpe Ratio": sharpe_ratio,
        "Cumulative Return": cumulative_return,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }

# Plot Portfolio Performance
def plot_performance(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['Portfolio_Value'], label='Portfolio Value', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Updated LSTM Trading Strategy: Portfolio Performance')
    plt.legend()
    plt.grid()
    plt.show()