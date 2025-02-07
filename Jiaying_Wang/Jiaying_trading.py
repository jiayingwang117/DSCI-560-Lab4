# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import mysql.connector

# # Database connection configuration
# DB_CONFIG = {
#     'host': 'localhost',
#     'user': 'root',  # or your MySQL username
#     'password': 'Ashley020501',
#     'database': 'stock_analysis'
# }

# def create_connection():
#     return mysql.connector.connect(**DB_CONFIG)

# # Fetch stock data for a given portfolio
# def fetch_portfolio_stocks(portfolio_id):
#     conn = create_connection()
#     query = """
#         SELECT ps.symbol, sd.date, sd.close_price 
#         FROM portfolio_stocks ps
#         JOIN stock_data sd ON ps.symbol = sd.symbol
#         WHERE ps.portfolio_id = %s
#         ORDER BY sd.date ASC
#     """
#     df = pd.read_sql(query, conn, params=(portfolio_id,))
#     conn.close()
#     df['date'] = pd.to_datetime(df['date'])
#     return df

# # Let user choose portfolio_id
# portfolio_id = int(input("Enter Portfolio ID: "))
# df = fetch_portfolio_stocks(portfolio_id)

# # Group by symbol for multi-stock analysis
# portfolio_value = {}
# initial_capital = 10000  # Initial investment amount per stock

# for symbol, stock_df in df.groupby('symbol'):
#     stock_df.set_index('date', inplace=True)
    
#     # Parameters for moving averages
#     short_window = 50
#     long_window = 200
    
#     # Calculate moving averages
#     stock_df['Short_MA'] = stock_df['close_price'].rolling(window=short_window, min_periods=1).mean()
#     stock_df['Long_MA'] = stock_df['close_price'].rolling(window=long_window, min_periods=1).mean()
    
#     # Generate buy/sell signals
#     stock_df['Signal'] = 0  # Default is no signal
#     stock_df.loc[stock_df['Short_MA'] > stock_df['Long_MA'], 'Signal'] = 1  # Buy
#     stock_df.loc[stock_df['Short_MA'] < stock_df['Long_MA'], 'Signal'] = -1  # Sell
    
#     # Implement trading strategy
#     shares_held = 0
#     cash = initial_capital
#     stock_portfolio_value = []
    
#     for i in range(len(stock_df)):
#         if stock_df.iloc[i]['Signal'] == 1 and cash > 0:  # Buy
#             shares_held = cash / stock_df.iloc[i]['close_price']
#             cash = 0
#         elif stock_df.iloc[i]['Signal'] == -1 and shares_held > 0:  # Sell
#             cash = shares_held * stock_df.iloc[i]['close_price']
#             shares_held = 0
#         stock_portfolio_value.append(cash + shares_held * stock_df.iloc[i]['close_price'])
    
#     stock_df['Portfolio_Value'] = stock_portfolio_value
#     portfolio_value[symbol] = stock_df

# # Combine portfolio values
# overall_portfolio_df = pd.concat(portfolio_value.values())

# # Plot results
# plt.figure(figsize=(12,6))
# for symbol, stock_df in portfolio_value.items():
#     plt.plot(stock_df.index, stock_df['Portfolio_Value'], label=f'Portfolio Value - {symbol}')
# plt.legend()
# plt.title('Portfolio Value Over Time')
# plt.show()

# # Performance Metrics
# final_value = overall_portfolio_df.groupby('date')['Portfolio_Value'].sum().iloc[-1]
# annualized_return = (final_value / (initial_capital * len(portfolio_value))) ** (252 / len(overall_portfolio_df.index.unique())) - 1
# sharpe_ratio = (overall_portfolio_df.groupby('date')['Portfolio_Value'].sum().pct_change().mean() / 
#                 overall_portfolio_df.groupby('date')['Portfolio_Value'].sum().pct_change().std()) * np.sqrt(252)

# print(f'Final Portfolio Value: ${final_value:.2f}')
# print(f'Annualized Return: {annualized_return:.2%}')
# print(f'Sharpe Ratio: {sharpe_ratio:.2f}')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Database connection configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',  # or your MySQL username
    'password': 'Ashley020501',
    'database': 'stock_analysis'
}

# Create SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

# Fetch stock data for a given portfolio
def fetch_portfolio_stocks(portfolio_id):
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

# Let user choose portfolio_id
portfolio_id = int(input("Enter Portfolio ID: "))
df = fetch_portfolio_stocks(portfolio_id)

# Group by symbol for multi-stock analysis
portfolio_value = {}
initial_capital = 10000  # Initial investment amount per stock

for symbol, stock_df in df.groupby('symbol'):
    stock_df.set_index('date', inplace=True)
    
    # Parameters for moving averages
    short_window = 50
    long_window = 200
    
    # Calculate moving averages
    stock_df['Short_MA'] = stock_df['close_price'].rolling(window=short_window, min_periods=1).mean()
    stock_df['Long_MA'] = stock_df['close_price'].rolling(window=long_window, min_periods=1).mean()
    
    # Generate buy/sell signals
    stock_df['Signal'] = 0  # Default is no signal
    stock_df.loc[stock_df['Short_MA'] > stock_df['Long_MA'], 'Signal'] = 1  # Buy
    stock_df.loc[stock_df['Short_MA'] < stock_df['Long_MA'], 'Signal'] = -1  # Sell
    
    # Implement trading strategy
    shares_held = 0
    cash = initial_capital
    stock_portfolio_value = []
    
    for i in range(len(stock_df)):
        if stock_df.iloc[i]['Signal'] == 1 and cash > 0:  # Buy
            shares_held = cash / stock_df.iloc[i]['close_price']
            cash = 0
        elif stock_df.iloc[i]['Signal'] == -1 and shares_held > 0:  # Sell
            cash = shares_held * stock_df.iloc[i]['close_price']
            shares_held = 0
        stock_portfolio_value.append(cash + shares_held * stock_df.iloc[i]['close_price'])
    
    stock_df['Portfolio_Value'] = stock_portfolio_value
    portfolio_value[symbol] = stock_df

# Combine portfolio values
overall_portfolio_df = pd.concat(portfolio_value.values())

# Plot results
plt.figure(figsize=(12,6))
for symbol, stock_df in portfolio_value.items():
    plt.plot(stock_df.index, stock_df['Portfolio_Value'], label=f'Portfolio Value - {symbol}')
plt.legend()
plt.title('Portfolio Value Over Time')
plt.show()

# Performance Metrics
final_value = overall_portfolio_df.groupby('date')['Portfolio_Value'].sum().iloc[-1]
annualized_return = (final_value / (initial_capital * len(portfolio_value))) ** (252 / len(overall_portfolio_df.index.unique())) - 1
sharpe_ratio = (overall_portfolio_df.groupby('date')['Portfolio_Value'].sum().pct_change().mean() / 
                overall_portfolio_df.groupby('date')['Portfolio_Value'].sum().pct_change().std()) * np.sqrt(252)

print(f'Final Portfolio Value: ${final_value:.2f}')
print(f'Annualized Return: {annualized_return:.2%}')
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
