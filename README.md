# DSCI-560-Lab4


## Jiaying_Wang Trading Algorithm Strategy Implementation
To implement Jiaying_Wang's trading algorithm strategy, navigate to the `Jiaying_Wang` folder. Inside, you will find `fetch.py`, which was copied from Lab 3 and follows the same instructions as in Lab 3. Before running any scripts, ensure that you have created the database as described in Lab 3. Once the database is set up, use `fetch.py` to populate portfolios and stock data as needed. After that, run `Jiaying_trading.py`, where you will be prompted to enter the **portfolio ID** you want to analyze. The script applies a **hybrid SMA + ARIMA trading strategy**, simulating trades with an initial value of **$10,000 per stock** while incorporating risk management rules. At the end of the simulation, it will output key performance metrics, including the **annualized return** and **Sharpe ratio**, along with a **graph showing portfolio value over time**, providing insights into the strategy’s effectiveness.

## Chu-Huan Huang Trading Algorithm Strategy
**1. Data Preparation:**

   - **Fetch Stock Data:**  Run `fetch.py` from the `Jiaying_Wang` folder (from Lab 3) to scrape historical stock data from Yahoo Finance and store it in your database. 
   - **Note Portfolio ID:**  `fetch.py` will display a **numerical Portfolio ID**.  Record this ID for the next step.

**2. Run Trading Simulation (`MovingAverage.py`):**

   - **Execute `MovingAverage.py` (in Chu-Huan Huang folder).**
   - **Enter Portfolio ID:** Input the Portfolio ID noted in the previous step when prompted.
   - **Strategy Choice:** Select a backtesting strategy:
      1. Single MA
      2. Dual MA
      3. Triple MA
   - **MA Methods:** Choose Moving Average methods (SMA, EMA, WMA) for the selected strategy.

**3. Performance Metrics Output:**

   - The simulation will output key performance metrics:
      - Portfolio Value
      - Annualized Return
      - Sharpe Ratio

## Chung-Sheng Chang Trading Algorithm Strategy
```
stock_prediction_project/
│── main.py                   # Main script to train & test LSTM
│── data_loader.py             # Fetches stock data & processes indicators
│── lstm_model.py              # Defines LSTM model
│── train.py                   # Training function with accuracy tracking
│── trading_simulation.py      # Mock trading strategy & portfolio visualization
│── config.py                  # Configurations (database, model parameters)
│── README.md                  # Project documentation
```
### Data Processing

**Fetching Stock Data from MySQL**
We extract historical stock price data from a MySQL database:
- Columns retrieved: date, open_price, high_price, low_price, close_price, volume
- The data is sorted by date for proper time-series processing.

**Feature Engineering (Technical Indicators)**

1.	Simple Moving Averages (SMA) – 10-day and 50-day moving averages.
2.	Exponential Moving Average (EMA) – 10-day EMA for short-term trend.
3.	Relative Strength Index (RSI) – Measures momentum to detect overbought/oversold conditions.
4.	Bollinger Bands (BB) – Measures market volatility.
5.	MACD (Moving Average Convergence Divergence) – Captures trend strength.

**Label Generation (Target Variable)**

We define the true label as:
- 1 → If stock increases after 5 days.
- 0 → If stock decreases after 5 days.

**Data Preparation for LSTM**
- Scaling Data
Normalize features using MinMaxScaler to improve LSTM training.
- Creating Time-Series Sequences
Convert data into 60-day lookback windows for training.

### Model Training

**Define LSTM Architecture**

The LSTM model consists of:
- 2 LSTM Layers (50 hidden units each).
- Fully Connected (Dense) Layer with Sigmoid activation.
- Binary Cross-Entropy Loss since this is a classification task.

**Training the Model**
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy (BCE)
- Batch Size: 32
- Epochs: 20

**Making Predictions**
- After training, we use the test dataset for predictions.
- Convert outputs into binary labels (0 or 1) using 0.5 threshold.

**Evaluating Performance (Mock Trading)**

Buying Condition
- If **LSTM predicts a price increase (`1`)** and cash is available → **Buy as many shares as possible**.
- Use **position sizing** (e.g., only invest **20%** of available capital per trade).

Selling Condition
- If **LSTM predicts a price drop (`0`)** and we own shares → **Sell all holdings**.
- Implement a **trailing stop-loss (2%)** to protect profits.

Holding Condition
- If **no new buy/sell signals** → **Maintain the current portfolio**.
**Risk Management Enhancements**
- **Stop-Loss Trigger**: Automatically **sell** if the price **drops 5% from the purchase price**.
- **Take-Profit Strategy**: Sell if the stock **gains 10%** after buying.
- **Fixed Position Sizing**: Only risk a fraction (e.g., **20%**) of capital per trade.
- **Minimum Holding Period**: Prevents frequent trades by **holding for at least 3 days** before selling.


### Results & Visualization

At the end, the script plots a graph showing:
- ✔ Portfolio Value Over Time
- ✔ Comparison with Stock Price Movements