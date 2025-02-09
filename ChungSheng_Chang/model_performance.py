import os
import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

load_dotenv()

# MySQL Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': 'stock_analysis'
}

# Fetch stock data from MySQL
def fetch_stock_data(symbol):
    conn = mysql.connector.connect(**DB_CONFIG)
    query = f"SELECT date, open_price, high_price, low_price, close_price, volume FROM stock_data WHERE symbol = '{symbol}' ORDER BY date"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Compute Technical Indicators
def compute_indicators(df):
    df['SMA_10'] = df['close_price'].rolling(window=10).mean()
    df['SMA_50'] = df['close_price'].rolling(window=50).mean()
    df['EMA_10'] = df['close_price'].ewm(span=10, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # Bollinger Bands
    df['BB_Middle'] = df['close_price'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + (df['close_price'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['close_price'].rolling(window=20).std() * 2)

    # MACD
    df['MACD'] = df['EMA_10'] - df['close_price'].ewm(span=26, adjust=False).mean()

    return df.dropna()

# Add True Labels (if stock price increases after 5 days)
def add_true_label(df):
    df['Future_Close'] = df['close_price'].shift(-5)
    df['Label'] = (df['Future_Close'] > df['close_price']).astype(int)
    df.drop(columns=['Future_Close'], inplace=True)
    return df.dropna()