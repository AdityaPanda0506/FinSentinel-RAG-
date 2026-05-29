"""
news_scraper.py
---------------
Scrapes financial news headlines from NewsAPI and stores
them in a local SQLite database, tagged by ticker.

Requires:
    NEWSAPI_KEY in your .env file
    pip install requests python-dotenv

Usage:
    python news_scraper.py              # scrape news for all tickers
    python news_scraper.py --ticker AAPL

Free tier limits:
    100 requests/day, headlines up to 1 month old.
    For historical data beyond 1 month → use Kaggle dataset instead.

Output:
    data/raw_news.db  →  table: news
"""

import sqlite3
import json
import time
import logging
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
TICKERS_FILE = Path(__file__).parent / "tickers.json"
DB_PATH      = DATA_DIR / "raw_news.db"

DATA_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
NEWSAPI_KEY      = os.getenv("NEWSAPI_KEY")
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"

# How many days back to scrape (free tier: max 30 days)
DAYS_BACK = 28

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Database setup ─────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT    NOT NULL,
            headline     TEXT    NOT NULL,
            description  TEXT,
            source       TEXT,
            url          TEXT,
            published_at TEXT,
            fetched_at   TEXT    NOT NULL,
            UNIQUE(ticker, url)           -- avoid duplicate articles
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_ticker ON news(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_date   ON news(published_at)")
    conn.commit()
    log.info("Database initialised → %s", DB_PATH)


# ── NewsAPI fetch ──────────────────────────────────────────────────────────────
def build_query(ticker: str, company_name: str) -> str:
    """
    Build a smart search query.
    Combines ticker symbol with company name for better recall.
    e.g.  'AAPL OR Apple stock'
    """
    return f'"{ticker}" OR "{company_name}" stock'


def fetch_news_for_ticker(
    ticker: str,
    company_name: str,
    days_back: int = DAYS_BACK,
) -> list[dict]:
    """
    Call NewsAPI and return a list of article dicts for this ticker.
    Handles pagination (up to 100 results per request on free tier).
    """
    if not NEWSAPI_KEY:
        raise EnvironmentError(
            "NEWSAPI_KEY not found. Add it to your .env file.\n"
            "Get a free key at: https://newsapi.org/register"
        )

    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date   = datetime.utcnow().strftime("%Y-%m-%d")
    query     = build_query(ticker, company_name)

    params = {
        "q":          query,
        "from":       from_date,
        "to":         to_date,
        "language":   "en",
        "sortBy":     "publishedAt",
        "pageSize":   100,           # max per request on free tier
        "page":       1,
        "apiKey":     NEWSAPI_KEY,
    }

    log.info("Scraping news for %-6s  (%s → %s)", ticker, from_date, to_date)

    try:
        resp = requests.get(NEWSAPI_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        log.error("NewsAPI request failed for %s: %s", ticker, exc)
        return []

    if data.get("status") != "ok":
        log.warning("NewsAPI error for %s: %s", ticker, data.get("message", "unknown"))
        return []

    articles = data.get("articles", [])
    log.info("  → %d articles returned for %s", len(articles), ticker)
    return articles


def parse_articles(ticker: str, articles: list[dict]) -> list[dict]:
    """
    Clean and normalise raw API response into flat dicts
    ready for SQLite insertion.
    """
    rows = []
    for art in articles:
        headline = art.get("title", "").strip()
        url      = art.get("url", "").strip()

        # Skip junk articles
        if not headline or not url:
            continue
        if headline.lower() in ("[removed]", ""):
            continue

        rows.append({
            "ticker":       ticker,
            "headline":     headline,
            "description":  (art.get("description") or "").strip()[:500],
            "source":       art.get("source", {}).get("name", ""),
            "url":          url,
            "published_at": art.get("publishedAt", "")[:10],  # keep date only
            "fetched_at":   datetime.utcnow().isoformat(),
        })
    return rows


def save_articles(rows: list[dict], conn: sqlite3.Connection) -> int:
    """
    Insert articles into the DB. Skips duplicates via UNIQUE constraint.
    Returns number of new rows inserted.
    """
    if not rows:
        return 0

    cursor     = conn.cursor()
    inserted   = 0

    for row in rows:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO news
                    (ticker, headline, description, source, url, published_at, fetched_at)
                VALUES
                    (:ticker, :headline, :description, :source, :url, :published_at, :fetched_at)
            """, row)
            inserted += cursor.rowcount
        except sqlite3.Error as exc:
            log.warning("DB insert error: %s", exc)

    conn.commit()
    return inserted


# ── Kaggle fallback loader ─────────────────────────────────────────────────────
def load_kaggle_csv(csv_path: str, conn: sqlite3.Connection) -> int:
    """
    Load a pre-downloaded Kaggle financial news CSV into the same DB.
    Expected columns: ticker, headline (or title), date (or published_at)

    Usage:
        from ingestion.news_scraper import load_kaggle_csv
        load_kaggle_csv("data/kaggle_financial_news.csv", conn)

    Kaggle dataset: 'Financial News and Stock Price Integration'
    """
    import pandas as pd

    log.info("Loading Kaggle CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Normalise common column name variants
    rename_map = {
        "title":        "headline",
        "text":         "headline",
        "date":         "published_at",
        "publishedAt":  "published_at",
        "company":      "ticker",
        "symbol":       "ticker",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"ticker", "headline", "published_at"}
    missing  = required - set(df.columns)
    if missing:
        log.error("CSV missing columns: %s. Available: %s", missing, list(df.columns))
        return 0

    df["description"] = df.get("description", "")
    df["source"]      = "kaggle"
    df["url"]         = df.get("url", df["headline"].str[:80])  # use headline as fallback url
    df["fetched_at"]  = datetime.utcnow().isoformat()

    df = df[["ticker", "headline", "description", "source", "url", "published_at", "fetched_at"]]
    df = df.dropna(subset=["headline"])
    df["headline"] = df["headline"].str.strip()
    df = df[df["headline"] != ""]

    inserted = 0
    cursor   = conn.cursor()
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO news
                    (ticker, headline, description, source, url, published_at, fetched_at)
                VALUES
                    (:ticker, :headline, :description, :source, :url, :published_at, :fetched_at)
            """, row.to_dict())
            inserted += cursor.rowcount
        except sqlite3.Error as exc:
            log.warning("DB insert error: %s", exc)

    conn.commit()
    log.info("Kaggle CSV → %d new rows inserted", inserted)
    return inserted


# ── Main orchestrator ──────────────────────────────────────────────────────────
def run(ticker_filter: str | None = None) -> None:
    with open(TICKERS_FILE) as f:
        config = json.load(f)

    tickers = config["watchlist"]

    if ticker_filter:
        tickers = [t for t in tickers if t["ticker"] == ticker_filter.upper()]
        if not tickers:
            log.error("Ticker %s not found in watchlist", ticker_filter)
            return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    total_inserted = 0

    for entry in tickers:
        ticker  = entry["ticker"]
        name    = entry["name"]

        articles = fetch_news_for_ticker(ticker, name)
        rows     = parse_articles(ticker, articles)
        inserted = save_articles(rows, conn)

        log.info("  ✓ %s  →  %d new articles saved", ticker, inserted)
        total_inserted += inserted

        # Be polite to the API — 1 request/second stays well within free limits
        time.sleep(1.2)

    conn.close()
    log.info("─" * 60)
    log.info("Done. Total new articles: %d", total_inserted)
    log.info("DB location: %s", DB_PATH)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape financial news from NewsAPI")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Scrape news for a single ticker (e.g. --ticker TSLA)")
    parser.add_argument("--kaggle", type=str, default=None,
                        help="Path to a Kaggle CSV file to load instead of calling NewsAPI")
    args = parser.parse_args()

    if args.kaggle:
        conn = sqlite3.connect(DB_PATH)
        init_db(conn)
        load_kaggle_csv(args.kaggle, conn)
        conn.close()
    else:
        run(ticker_filter=args.ticker)