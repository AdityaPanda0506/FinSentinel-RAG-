"""
price_fetcher.py
----------------
Pulls historical OHLCV stock price data using yfinance
and stores it in a local SQLite database.

Usage:
    python price_fetcher.py              # fetch all tickers in tickers.json
    python price_fetcher.py --ticker AAPL  # fetch a single ticker

Output:
    data/raw_prices.db  →  table: prices
"""

import sqlite3
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

import yfinance as yf
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
TICKERS_FILE = Path(__file__).parent / "tickers.json"
DB_PATH     = DATA_DIR / "raw_prices.db"

DATA_DIR.mkdir(exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Database setup ─────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    """Create the prices table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            fetched_at  TEXT    NOT NULL,
            UNIQUE(ticker, date)          -- avoid duplicate rows on re-run
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date   ON prices(date)")
    conn.commit()
    log.info("Database initialised → %s", DB_PATH)


# ── Core fetch ─────────────────────────────────────────────────────────────────
def fetch_ticker(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV data for one ticker from Yahoo Finance.
    Returns a clean DataFrame with columns:
        ticker, date, open, high, low, close, volume
    """
    log.info("Fetching %-6s  %s → %s", ticker, start, end)
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,   # adjusts for splits & dividends
            progress=False,
        )
    except Exception as exc:
        log.error("yfinance error for %s: %s", ticker, exc)
        return pd.DataFrame()

    if df.empty:
        log.warning("No data returned for %s", ticker)
        return pd.DataFrame()

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    # Rename 'price date' column yfinance sometimes returns
    if "price date" in df.columns:
        df = df.rename(columns={"price date": "date"})

    df["ticker"] = ticker
    df["date"]   = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Keep only what we need
    df = df[["ticker", "date", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["close"])

    log.info("  → %d rows for %s", len(df), ticker)
    return df


def save_to_db(df: pd.DataFrame, conn: sqlite3.Connection) -> int:
    """
    Insert rows into the prices table using INSERT OR IGNORE.
    Safe to run multiple times — skips duplicates automatically.
    Returns number of new rows inserted.
    """
    if df.empty:
        return 0

    import datetime as dt
    fetched_at = datetime.now(dt.timezone.utc).isoformat()
    df = df.copy()
    df["fetched_at"] = fetched_at

    cursor   = conn.cursor()
    inserted = 0

    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO prices
                    (ticker, date, open, high, low, close, volume, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["ticker"], row["date"], row["open"],
                row["high"],   row["low"],  row["close"],
                row["volume"], row["fetched_at"],
            ))
            inserted += cursor.rowcount
        except Exception:
            pass

    conn.commit()
    return inserted


# ── Main orchestrator ──────────────────────────────────────────────────────────
def run(ticker_filter: str | None = None) -> None:
    with open(TICKERS_FILE) as f:
        config = json.load(f)

    tickers  = [t["ticker"] for t in config["watchlist"]]
    settings = config["settings"]
    start    = settings["historical_start"]
    end      = settings["historical_end"]
    interval = settings["interval"]

    if ticker_filter:
        tickers = [t for t in tickers if t == ticker_filter.upper()]
        if not tickers:
            log.error("Ticker %s not found in watchlist", ticker_filter)
            return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    total_inserted = 0
    failed = []

    for ticker in tickers:
        df = fetch_ticker(ticker, start, end, interval)
        if df.empty:
            failed.append(ticker)
            continue

        inserted = save_to_db(df, conn)
        total_inserted += inserted
        log.info("  ✓ %s  →  %d new rows saved", ticker, inserted)

        time.sleep(0.3)  # gentle rate limiting — Yahoo Finance doesn't love hammering

    conn.close()

    log.info("─" * 60)
    log.info("Done. Total new rows: %d", total_inserted)
    if failed:
        log.warning("Failed tickers: %s", ", ".join(failed))


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock price data from Yahoo Finance")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Fetch a single ticker (e.g. --ticker AAPL). Omit to fetch all.")
    args = parser.parse_args()
    run(ticker_filter=args.ticker)