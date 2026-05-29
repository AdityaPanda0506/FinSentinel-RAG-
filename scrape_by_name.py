import requests, sqlite3, time, os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
API_KEY = os.getenv("NEWSAPI_KEY")
DB_PATH = Path("data/raw_news.db")

# Company name → ticker mapping for better search results
SEARCHES = {
    "AAPL":  ["Apple stock earnings", "Apple Inc AAPL"],
    "MSFT":  ["Microsoft stock", "Microsoft earnings"],
    "GOOGL": ["Google Alphabet stock", "Alphabet earnings"],
    "AMZN":  ["Amazon stock", "Amazon earnings"],
    "TSLA":  ["Tesla stock", "Tesla earnings Elon"],
    "NVDA":  ["NVIDIA stock", "NVIDIA earnings GPU"],
    "META":  ["Meta Facebook stock", "Meta earnings"],
    "JPM":   ["JPMorgan stock", "JPMorgan Chase earnings"],
    "BAC":   ["Bank of America stock", "BAC earnings"],
    "GS":    ["Goldman Sachs stock", "Goldman earnings"],
    "JNJ":   ["Johnson Johnson stock", "JNJ earnings"],
    "PFE":   ["Pfizer stock", "Pfizer earnings"],
    "TSLA":  ["Tesla stock price", "Tesla deliveries"],
    "XOM":   ["ExxonMobil stock", "Exxon earnings oil"],
    "WMT":   ["Walmart stock", "Walmart earnings retail"],
    "NVDA":  ["NVIDIA GPU AI stock", "NVIDIA quarterly results"],
    "AMD":   ["AMD chip stock", "AMD earnings results"],
    "NFLX":  ["Netflix stock subscribers", "Netflix earnings"],
    "DIS":   ["Disney stock earnings", "Walt Disney results"],
    "PYPL":  ["PayPal stock", "PayPal earnings fintech"],
}

def fetch_articles(query, from_date, api_key):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": 100,
        "apiKey":   api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("status") == "ok":
            return data.get("articles", [])
    except Exception as e:
        print(f"Error: {e}")
    return []

def save_articles(ticker, articles, conn):
    inserted = 0
    for art in articles:
        headline = art.get("title", "").strip()
        url      = art.get("url", "").strip()
        date     = art.get("publishedAt", "")[:10]
        if not headline or not url or headline == "[Removed]":
            continue
        try:
            conn.execute("""
                INSERT OR IGNORE INTO news
                (ticker, headline, description, source, url, published_at, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, headline, "", 
                  art.get("source", {}).get("name", ""),
                  url, date, datetime.utcnow().isoformat()))
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except:
            pass
    conn.commit()
    return inserted

# Run
conn = sqlite3.connect(DB_PATH)
from_date = (datetime.utcnow() - timedelta(days=29)).strftime("%Y-%m-%d")

total = 0
seen_queries = set()
for ticker, queries in SEARCHES.items():
    for query in queries:
        if query in seen_queries:
            continue
        seen_queries.add(query)
        print(f"Fetching: {query}")
        articles = fetch_articles(query, from_date, API_KEY)
        n = save_articles(ticker, articles, conn)
        total += n
        print(f"  → {len(articles)} fetched, {n} new saved for {ticker}")
        time.sleep(1.2)

conn.close()
print(f"\nDone. Total new articles: {total}")