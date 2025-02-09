import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import matplotlib.colors as mcolors

load_dotenv()

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'user': os.environ.get('DB_USER', 'root'),
    'password': os.environ.get('DB_PASSWORD'),
    'database': os.environ.get('DB_DATABASE', 'stock_analysis')
}

# Database engine setup
db_engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

def get_stock_data(portfolio_id):
    """
    Fetches stock data for a given portfolio ID from the database.
    """
    query = """
        SELECT ps.symbol, sd.date, sd.close_price
        FROM portfolio_stocks ps
        JOIN stock_data sd ON ps.symbol = sd.symbol
        WHERE ps.portfolio_id = %s
        ORDER BY sd.date ASC
    """
    
    df = pd.read_sql(query, db_engine, params=(portfolio_id,))
    df['date'] = pd.to_datetime(df['date'])
    return df

def compute_moving_averages(df, periods, methods=['SMA']):
    """
    Computes Simple Moving Average (SMA), Exponential Moving Average (EMA),
    and Weighted Moving Average (WMA) for the specified periods and methods.
    """
    for method in methods:
        for period in periods:
            col_name = f'{method}_{period}'
            if method == 'SMA':
                df[col_name] = df['close_price'].rolling(window=period, min_periods=1).mean()
            elif method == 'EMA':
                df[col_name] = df['close_price'].ewm(span=period, adjust=False).mean()
            elif method == 'WMA':
                weights = np.arange(1, period + 1)
                df[col_name] = df['close_price'].rolling(window=period).apply(
                    lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return df

def generate_signals_triple_ma(df, short_term=20, medium_term=50, long_term=200, method='SMA'):
    """
    Generates trading signals based on a Triple Moving Average Crossover strategy.
    
    Uses a single method (default 'SMA') to compute three moving averages:
      - Buy signal (1): when short-term MA > medium-term MA > long-term MA
      - Sell signal (-1): when short-term MA < medium-term MA < long-term MA
      - Otherwise, hold (0)
    """
    # For visualization, we also compute additional periods (e.g., 5 and 20)
    periods = sorted([short_term, medium_term, long_term, 5, 20])
    df = compute_moving_averages(df, periods, methods=[method])
    df['Signal'] = 0

    short_ma_col = f'{method}_{short_term}'
    medium_ma_col = f'{method}_{medium_term}'
    long_ma_col = f'{method}_{long_term}'

    # Buy signal: short MA above medium MA and medium MA above long MA
    df.loc[(df[short_ma_col] > df[medium_ma_col]) & (df[medium_ma_col] > df[long_ma_col]), 'Signal'] = 1

    # Sell signal: short MA below medium MA and medium MA below long MA
    df.loc[(df[short_ma_col] < df[medium_ma_col]) & (df[medium_ma_col] < df[long_ma_col]), 'Signal'] = -1

    return df

def generate_signals(df, short_term=10, long_term=60, method='SMA'):
    """
    Generates trading signals based on a Dual Moving Average Crossover strategy.
    
    Uses a single method (default 'SMA'):
      - Buy signal (1): when short-term MA > long-term MA
      - Sell signal (-1): when short-term MA < long-term MA
      - Otherwise, hold (0)
    """
    # Compute the short and long moving averages along with additional periods for visualization
    df = compute_moving_averages(df, [short_term, long_term, 5, 20], methods=[method])
    df['Signal'] = 0

    # Use only one method's moving averages for signal generation
    short_ma_col = f'{method}_{short_term}'
    long_ma_col = f'{method}_{long_term}'

    df.loc[df[short_ma_col] > df[long_ma_col], 'Signal'] = 1
    df.loc[df[short_ma_col] < df[long_ma_col], 'Signal'] = -1
    return df

def generate_signals_single_ma(df, period=20, method='SMA'):
    """
    Generates trading signals based on a Single Moving Average Crossover strategy.
    
    Uses a single moving average:
      - Buy signal (1): when close price is above the MA
      - Sell signal (-1): when close price is below the MA
      - Otherwise, hold (0)
    """
    # Compute the specified moving average (plus additional periods for visualization)
    df = compute_moving_averages(df, [period, 5, 20], methods=[method])
    ma_col_name = f'{method}_{period}'
    df['Signal'] = 0

    df.loc[df['close_price'] > df[ma_col_name], 'Signal'] = 1
    df.loc[df['close_price'] < df[ma_col_name], 'Signal'] = -1

    return df


def execute_trades_with_commission_tp_sl(df, capital, shares, commission_percentage=0.0005, take_profit_percent=0.15, stop_loss_percent=0.07):
    """
    Executes trading strategy with commission, take profit and stop loss orders.
    """
    # Use passed capital and shares, rename shares to current_shares to avoid confusion
    cash, current_shares = capital, shares
    
    df['Portfolio_Value'] = 0.0
    df['Cash_Remaining'] = 0.0
    df['Shares_Held'] = 0.0
    df['Entry_Price'] = 0.0
    df['Commission_Paid'] = 0.0
    df['Position'] = 0 

    # Using integer-based indexing with iloc for performance and clarity
    for i in range(len(df)): 

        price = df.iloc[i]['close_price'] # Current day's closing price
        signal = df.iloc[i]['Signal']     # Trading signal for the current day


        # Check for Take Profit or Stop Loss conditions before new signals
        # If in a long position from the previous day and not already selling
        if i > 0 and df.iloc[i-1]['Position'] == 1 and current_shares > 0: 
            entry_price = df.iloc[i-1]['Entry_Price'] # Get entry price from previous day
            take_profit_price = entry_price * (1 + take_profit_percent) 
            stop_loss_price = entry_price * (1 - stop_loss_percent)    

            if price >= take_profit_price: # Take Profit Condition met
                cash_before_commission = current_shares * price
                commission = cash_before_commission * commission_percentage
                cash = cash_before_commission - commission
                current_shares = 0 # Exit position
                df.iloc[i, df.columns.get_loc('Signal')] = -1  
                df.iloc[i, df.columns.get_loc('Commission_Paid')] += commission 
                df.iloc[i, df.columns.get_loc('Position')] = 0 
                print(f"{df.iloc[i].name.strftime('%Y-%m-%d')}: {df.iloc[i]['symbol']} Take Profit at {take_profit_price:.2f}, Selling (Commission: ${commission:.2f}).")
                continue 

            elif price <= stop_loss_price: # Stop Loss Condition met
                cash_before_commission = current_shares * price
                commission = cash_before_commission * commission_percentage
                cash = cash_before_commission - commission
                current_shares = 0 # Exit position
                df.iloc[i, df.columns.get_loc('Signal')] = -1  
                df.iloc[i, df.columns.get_loc('Commission_Paid')] += commission 
                df.iloc[i, df.columns.get_loc('Position')] = 0 
                print(f"{df.iloc[i].name.strftime('%Y-%m-%d')}: {df.iloc[i]['symbol']} Stop Loss at {stop_loss_price:.2f}, Selling (Commission: ${commission:.2f}).")
                continue 

        # Original Signal-Based Trading Logic
        if signal == 1 and cash > 0 and current_shares == 0: # Buy signal and have cash to buy and not already holding shares
            # Invest all available cash
            trade_amount = cash 
            commission = trade_amount * commission_percentage
            
            # Ensure trade amount is greater than commission
            if trade_amount > commission: 
                current_shares = (trade_amount - commission) / price # Calculate number of shares to buy
                cash = 0 # Cash is used to buy shares
                df.iloc[i, df.columns.get_loc('Entry_Price')] = price # Record entry price
                df.iloc[i, df.columns.get_loc('Commission_Paid')] += commission # Record commission paid
                df.iloc[i, df.columns.get_loc('Position')] = 1 # Enter long position
                print(f"{df.iloc[i].name.strftime('%Y-%m-%d')}: {df.iloc[i]['symbol']} Moving Average Buy Signal, Buying (Commission: ${commission:.2f}).")

        
        elif signal == -1 and current_shares > 0: 
            cash_before_commission = current_shares * price
            commission = cash_before_commission * commission_percentage
            cash = cash_before_commission - commission 
            current_shares = 0 
            df.iloc[i, df.columns.get_loc('Commission_Paid')] += commission 
            df.iloc[i, df.columns.get_loc('Position')] = 0 
            print(f"{df.iloc[i].name.strftime('%Y-%m-%d')}: {df.iloc[i]['symbol']} Moving Average Sell Signal, Selling (Commission: ${commission:.2f}).")

        # Update portfolio value, remaining cash, and shares held for the day
        df.iloc[i, df.columns.get_loc('Portfolio_Value')] = float(cash + (current_shares * price)) 
        df.iloc[i, df.columns.get_loc('Cash_Remaining')] = float(cash) 
        df.iloc[i, df.columns.get_loc('Shares_Held')] = float(current_shares) 

    return df, cash, current_shares


def calculate_performance(df):
    """
    Calculates portfolio performance metrics: Total Return, Annualized Return, and Sharpe Ratio.
    """
    df['Daily_Return'] = df['Portfolio_Value'].pct_change().fillna(0).astype(np.float64)
    df['Daily_Return'] = df['Daily_Return'].replace([np.inf, -np.inf], np.nan)
    daily_returns_no_nan = df['Daily_Return'].dropna()

    if daily_returns_no_nan.empty:
        print("Warning: No valid daily returns after dropna. Sharpe Ratio will be NaN.")
        sharpe_ratio = np.nan
    else:
        std_dev = daily_returns_no_nan.std()
        if std_dev <= 1e-9:
            print("Warning: Standard deviation is close to zero. Clamping to a small value to calculate Sharpe Ratio.")
            std_dev = max(std_dev, 1e-9)

        if np.isnan(std_dev):
            print("Warning: Standard deviation calculation resulted in NaN. Sharpe Ratio will be NaN.")
            sharpe_ratio = np.nan
        else:
            sharpe_ratio = daily_returns_no_nan.mean() / std_dev * np.sqrt(252)

    total_return = (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(df)) - 1

    return total_return, annualized_return, sharpe_ratio, df['Portfolio_Value'].iloc[-1]

def visualize_stock(df, symbol, ma_type='All', strategy_type='DualMA'):
    """
    Visualizes stock price and moving averages.
    """
    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")

    sns.lineplot(x=df.index, y=df['close_price'], label='Close Price', linestyle='-')

    if ma_type == 'All':
        ma_methods = ['SMA', 'EMA', 'WMA']
    else:
        ma_methods = [ma_type]

    colors = ['tab:orange', 'tab:red', 'tab:green']
    color_index = 0

    for method in ma_methods:
        for col in df.columns:
            if method in col and 'Signal' not in col and 'Portfolio_Value' not in col and 'Cash_Remaining' not in col and 'Shares_Held' not in col and 'Entry_Price' not in col and 'Commission_Paid' not in col and 'Daily_Return' not in col and method != 'WMA_5' and method != 'EMA_5' and method != 'SMA_5' and method != 'WMA_20' and method != 'EMA_20' and method != 'SMA_20':

                if strategy_type == 'TripleMA':
                    periods_to_plot = [20, 50, 200]
                elif strategy_type == 'SingleMA':
                    periods_to_plot = [20]
                else: # DualMA
                    periods_to_plot = [10, 60]

                for period in periods_to_plot:
                    if f'_{period}' in col and method in col:
                        sns.lineplot(x=df.index, y=df[col], label=col, color=colors[color_index % len(colors)], linestyle='--')
                        color_index += 1
                        # Avoid duplicate plotting
                        break 
                    
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.title(f"Stock Performance - {symbol} - with {ma_type} Moving Averages - Strategy: {strategy_type}")
    plt.legend()
    plt.show()

def main():
    """
    Main function to run the stock trading strategy backtesting.
    Gets user inputs for portfolio ID, strategy choice, and moving average methods,
    fetches stock data, generates trading signals, executes trades, calculates performance,
    and visualizes results.
    """
    try:
        portfolio_id = int(input("Enter Portfolio ID: "))
        strategy_choice = input("Choose strategy (1: Single MA, 2: Dual MA Crossover, 3: Triple MA Crossover): ")
        ma_methods_input = input("Enter MA methods to use (SMA, EMA, WMA, or All - comma separated, default: SMA,EMA): ").strip().upper()

        if not ma_methods_input:
            ma_methods = ['SMA', 'EMA']
        elif ma_methods_input.upper() == 'ALL':
            ma_methods = ['SMA', 'EMA', 'WMA']
        else:
            ma_methods = [m.strip() for m in ma_methods_input.split(',') if m.strip() in ['SMA', 'EMA', 'WMA']]
            if not ma_methods:
                print("Invalid MA methods input. Using default: SMA, EMA")
                ma_methods = ['SMA', 'EMA']

    except ValueError:
        print("Invalid input. Please enter a numeric ID for Portfolio ID and 1, 2, or 3 for strategy choice.")
        return

    data = get_stock_data(portfolio_id)
    if data.empty:
        print(f"No data found for portfolio ID: {portfolio_id}")
        return

    processed_results = []
    individual_stock_performance = {}
    
    # Initial capital and shares(can be modified based on user input)
    cash = 10000.0
    shares = 0.0

    for symbol in data['symbol'].unique():
        stock_df = data[data['symbol'] == symbol].copy()
        stock_df.set_index('date', inplace=True)
        stock_df.sort_index(inplace=True)

        if strategy_choice == "1": # Single MA 
            stock_df = generate_signals_single_ma(stock_df, period=20, method=ma_methods[0] if ma_methods else 'SMA')
            strategy_name = 'SingleMA'
            periods_visualize = [20]
            
        elif strategy_choice == "2": # Dual MA 
            stock_df = generate_signals(stock_df, short_term=10, long_term=60, methods=ma_methods)
            strategy_name = 'DualMA'
            periods_visualize = [10, 60]
            
        elif strategy_choice == "3": # Triple MA 
            stock_df = generate_signals_triple_ma(stock_df, short_term=20, medium_term=50, long_term=200, methods=ma_methods)
            strategy_name = 'TripleMA'
            periods_visualize = [20, 50, 200]
            
        else: # Default: Dual MA 
            stock_df = generate_signals(stock_df, short_term=10, long_term=60, methods=ma_methods)
            strategy_name = 'DualMA'
            periods_visualize = [10, 60]
            print("Invalid strategy choice. Defaulting to Dual MA Crossover Strategy.")

        stock_df, cash, shares = execute_trades_with_commission_tp_sl(
            stock_df,
            capital=cash, # Current cash
            shares=shares, # Current shares
            commission_percentage=0.0005,
            take_profit_percent=0.15,
            stop_loss_percent=0.07
        )
        
        for ma_type in ma_methods + ['All']:
            visualize_stock(stock_df, symbol, ma_type=ma_type, strategy_type=strategy_name)

        total_return, annualized_return, sharpe_ratio, final_portfolio_value = calculate_performance(stock_df)
        individual_stock_performance[symbol] = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Sharpe Ratio': sharpe_ratio,
            'Final Portfolio Value': final_portfolio_value
        }
        processed_results.append(stock_df)

    if processed_results:
        combined_df = pd.concat(processed_results)
        portfolio_total_return, portfolio_annualized_return, portfolio_sharpe_ratio, portfolio_final_value = calculate_performance(combined_df)

        print("\n=== Individual Stock Performance ===")
        for symbol, performance in individual_stock_performance.items():
            print(f"\n--- {symbol} ---")
            print(f"  Total Return: {performance['Total Return']:.2%}")
            print(f"  Annualized Return: {performance['Annualized Return']:.2%}")
            print(f"  Sharpe Ratio: {performance['Sharpe Ratio']:.2f}")
            print(f"  Final Portfolio Value: ${performance['Final Portfolio Value']:.2f}")


        print("\n=== Overall Portfolio Performance (Combined Stocks) ===")
        print(f"Total Portfolio Value: ${portfolio_final_value:.2f}")
        print(f"Annualized Return: {portfolio_annualized_return:.2%}")
        print(f"Sharpe Ratio: {portfolio_sharpe_ratio:.2f}")

    else:
        print("No stock data processed.")
        return

if __name__ == "__main__":
    main()
