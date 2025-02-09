import numpy as np
import pandas as pd
import torch
import mysql.connector
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from config import DB_CONFIG, LOOKBACK, BATCH_SIZE

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

# Custom Dataset Class for PyTorch
class StockDataset(Dataset):
    def __init__(self, data):
        self.features = data[['SMA_10', 'SMA_50', 'EMA_10', 'RSI', 'BB_Upper', 'BB_Lower', 'MACD']].values
        self.labels = data['Label'].values

        scaler = MinMaxScaler()
        self.features = scaler.fit_transform(self.features)

        self.X, self.y = [], []
        for i in range(len(self.features) - LOOKBACK):
            self.X.append(self.features[i:i + LOOKBACK])
            self.y.append(self.labels[i + LOOKBACK])

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# Create DataLoader
def create_dataloader(df):
    dataset = StockDataset(df)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)