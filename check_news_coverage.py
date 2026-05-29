import sqlite3
import pandas as pd

conn = sqlite3.connect('data/raw_news.db')

# Articles per ticker per year
df = pd.read_sql("""
    SELECT 
        ticker,
        substr(published_at, 1, 4) as year,
        COUNT(*) as articles
    FROM news
    GROUP BY ticker, year
    ORDER BY ticker, year
""", conn)

# Filter to your watchlist tickers
watchlist = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","JPM",
             "BAC","GS","JNJ","PFE","UNH","XOM","CVX","WMT","PG","KO",
             "DIS","NFLX","AMD","INTC","BA","CAT","V","MA","PYPL","CRM",
             "ADBE","ORCL","CSCO","QCOM","HD","MCD","NKE","SBUX"]

df = df[df['ticker'].isin(watchlist)]
pivot = df.pivot_table(index='ticker', columns='year', 
                        values='articles', fill_value=0)
print("Articles per ticker per year:")
print(pivot.to_string())
print()
print("Total articles per year:")
print(df.groupby('year')['articles'].sum())

conn.close()