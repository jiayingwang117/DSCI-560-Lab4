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
