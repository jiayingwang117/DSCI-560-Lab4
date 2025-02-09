import os
import mysql.connector
import yfinance as yf
import pandas as pd
import datetime
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',    
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': 'stock_analysis'
}

def create_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_portfolio_id_by_name(portfolio_name):
    """
    Returns the portfolio_id if the portfolio_name exists, else None.
    """
    conn = create_connection()
    cursor = conn.cursor()
    sql = "SELECT portfolio_id FROM portfolios WHERE portfolio_name = %s"
    cursor.execute(sql, (portfolio_name,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row[0] if row else None

def create_portfolio(portfolio_name):
    """
    Inserts a new portfolio into the portfolios table.
    Returns the newly generated or existing portfolio_id.
    """
    existing_portfolio_id = get_portfolio_id_by_name(portfolio_name)
    if existing_portfolio_id:
        # Portfolio already exists
        print(f"Portfolio '{portfolio_name}' already exists with ID = {existing_portfolio_id}.")
        return existing_portfolio_id
    else:
        # Create new portfolio
        conn = create_connection()
        cursor = conn.cursor()
        sql = "INSERT INTO portfolios (portfolio_name) VALUES (%s)"
        cursor.execute(sql, (portfolio_name,))
        conn.commit()
        portfolio_id = cursor.lastrowid
        cursor.close()
        conn.close()
        print(f"Created portfolio '{portfolio_name}' with ID = {portfolio_id}.")
        return portfolio_id

def add_stock_to_portfolio(portfolio_id, symbol):
    """
    Inserts a (portfolio_id, symbol) pair into portfolio_stocks.
    """
    conn = create_connection()
    cursor = conn.cursor()
    try:
        # Check if stock already exists in the portfolio
        check_sql = "SELECT 1 FROM portfolio_stocks WHERE portfolio_id = %s AND symbol = %s"
        cursor.execute(check_sql, (portfolio_id, symbol.upper()))
        exists = cursor.fetchone()

        if not exists:
            sql = "INSERT INTO portfolio_stocks (portfolio_id, symbol) VALUES (%s, %s)"
            cursor.execute(sql, (portfolio_id, symbol.upper()))
            conn.commit()
            print(f"'{symbol}' added to portfolio ID {portfolio_id}.")
        else:
            print(f"'{symbol}' already exists in portfolio ID {portfolio_id}, skipping.")

    except mysql.connector.Error as err:
        # Possibly a duplicate (portfolio_id, symbol)
        print(f"Error adding {symbol}: {err}")
    finally:
        cursor.close()
        conn.close()

def is_valid_symbol(symbol):
    """
    Checks if the given symbol has data on Yahoo Finance in the last 30 days.
    Returns True if we found data, False otherwise.
    """
    today = datetime.date.today()
    start_check = today - datetime.timedelta(days=30)

    df_check = yf.download(symbol, start=start_check, end=today, progress=False)
    return not df_check.empty

def fetch_and_store_stock_data(symbol, start_date, end_date):
    """
    Fetch stock data for the given symbol and date range using yfinance,
    then do an upsert (insert or overwrite) for each (symbol, date).
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        print(f"No data found for {symbol} in the given date range.")
        return

    df.reset_index(inplace=True)
    
    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    conn = create_connection()
    cursor = conn.cursor()
    # Use ON DUPLICATE KEY UPDATE to overwrite data if (symbol, date) already exists
    insert_sql = """
        INSERT INTO stock_data
        (symbol, date, open_price, high_price, low_price, close_price, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          open_price = VALUES(open_price),
          high_price = VALUES(high_price),
          low_price  = VALUES(low_price),
          close_price= VALUES(close_price),
          volume     = VALUES(volume)
    """

    count_inserted = 0

    for _, row in df.iterrows():
        # print(row)
        try:
            data_tuple = (
                symbol,
                row['Date'].date(),                     # Directly access the date part
                float(row[f'Open {symbol}']),
                float(row[f'High {symbol}']),
                float(row[f'Low {symbol}']),
                float(row[f'Close {symbol}']),
                int(row[f'Volume {symbol}'])
            )
            cursor.execute(insert_sql, data_tuple)
            count_inserted += 1
        except KeyError as e:
            print(f"Missing column in data: {e}. Skipping this row.")
        except Exception as e:
            print(f"Error processing row: {e}. Skipping this row.")

    conn.commit()
    print(f"Successfully Upserted {count_inserted} rows of data for symbol {symbol}.")
    cursor.close()
    conn.close()



def main():
    # Simple menu
    print("Welcome to the Stock Manager!")
    print("1) Create or reuse a portfolio and add stocks")
    print("2) Exit (do nothing)")
    choice = input("Enter your choice (1 or 2): ")

    if choice.strip() == "1":
        # Get or create portfolio
        portfolio_name = input("Enter a name for your portfolio: ")
        if not portfolio_name.strip():
            print("No portfolio name entered. Exiting.")
            return
        
        portfolio_id = create_portfolio(portfolio_name)

        stock_list = input("Enter stock symbols (comma-separated): ")
        symbols = [s.strip().upper() for s in stock_list.split(',') if s.strip()]

        if not symbols:
            print("No symbols entered. Exiting.")
            return

        # Validate each symbol before adding
        valid_symbols = []
        for symbol in symbols:
            if is_valid_symbol(symbol):
                add_stock_to_portfolio(portfolio_id, symbol)
                valid_symbols.append(symbol)
            else:
                print(f"'{symbol}' is invalid or has no recent data. Skipping.")

        if not valid_symbols:
            print("No valid symbols were added to the portfolio. Exiting.")
            return

        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")

        for symbol in valid_symbols:
            fetch_and_store_stock_data(symbol, start_date, end_date)

        print("Data collection complete.")
    else:
        print("No action taken. Goodbye!")
        return

if __name__ == "__main__":
    main()
