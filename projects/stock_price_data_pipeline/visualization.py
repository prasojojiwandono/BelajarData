import streamlit as st
import pandas as pd
import sqlite3

conn = sqlite3.connect("stocks.db")
df = pd.read_sql("SELECT * FROM aapl_prices", conn)

st.line_chart(df[['Close']])
