import yfinance as yf
import pandas as pd
import os
import sqlite3

file_path = 'hist.pkl'
if os.path.exists(file_path):
    hist = pd.read_pickle(file_path)
else:
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="1mo", interval="1h")
    pd.to_pickle(hist,'hist.pkl')


hist.dropna(inplace=True)
hist['return'] = hist['Close'].pct_change()
hist['sma_10'] = hist['Close'].rolling(window=10).mean()


conn = sqlite3.connect("stocks.db")
hist.to_sql("aapl_prices", conn, if_exists="replace")