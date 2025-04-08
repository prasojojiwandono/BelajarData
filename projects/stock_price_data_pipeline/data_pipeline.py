from prefect import flow, task
import sqlite3
import yfinance as yf

@task
def fetch_data():
    return yf.Ticker("AAPL").history(period="1d", interval="1h")

@task
def process_data(df):
    df['return'] = df['Close'].pct_change()
    return df

@task
def store_data(df):
    conn = sqlite3.connect("stocks.db")
    df.to_sql("aapl_prices", conn, if_exists="append")

@flow
def stock_pipeline():
    raw = fetch_data()
    processed = process_data(raw)
    store_data(processed)

stock_pipeline()
