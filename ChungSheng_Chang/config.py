import os
from dotenv import load_dotenv

load_dotenv()


# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': 'stock_analysis'
}

# LSTM Model Configuration
LOOKBACK = 60
INPUT_SIZE = 7  # Number of features
HIDDEN_SIZE = 50
NUM_LAYERS = 2
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001