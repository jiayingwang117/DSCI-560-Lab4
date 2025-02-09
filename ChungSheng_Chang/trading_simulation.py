import matplotlib.pyplot as plt

# Simulate Trading
def simulate_trading(df, y_pred, initial_balance=10000):
    df = df.iloc[len(df) - len(y_pred):].copy()
    df['Prediction'] = y_pred
    balance = initial_balance
    shares = 0
    portfolio_value = []

    for index, row in df.iterrows():
        if row['Prediction'] == 1 and balance >= row['close_price']:
            shares = balance / row['close_price']
            balance = 0
        elif row['Prediction'] == 0 and shares > 0:
            balance = shares * row['close_price']
            shares = 0
        portfolio_value.append(balance + (shares * row['close_price']))

    df['Portfolio_Value'] = portfolio_value
    return df

# Plot Performance
def plot_performance(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['Portfolio_Value'], label='Portfolio Value', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Mock Trading Portfolio Performance')
    plt.legend()
    plt.grid()
    plt.show()