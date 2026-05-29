# ingestion/stocktwits_scraper.py
import sqlite3, requests, time, logging
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = ROOT_DIR / "data" / "raw_news.db"

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s")
log = logging.getLogger(__name__)

def scrape_ticker(ticker: str, conn: sqlite3.Connection) -> int:
    url  = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception as e:
        log.error("Failed %s: %s", ticker, e)
        return 0

    messages = data.get("messages", [])
    inserted = 0
    for msg in messages:
        body = msg.get("body", "").strip()
        if not body:
            continue
        try:
            conn.execute("""
                INSERT OR IGNORE INTO news
                (ticker, headline, description, source, url, published_at, fetched_at)
                VALUES (?, ?, '', 'stocktwits', '', ?, ?)
            """, (
                ticker, body[:500],
                msg.get("created_at", "")[:10],
                datetime.utcnow().isoformat()
            ))
            inserted += conn.execute("SELECT changes()").fetchone()[0]
        except Exception:
            pass
    conn.commit()
    log.info("%-6s  →  %d new posts", ticker, inserted)
    return inserted

def run():
    import json
    tickers_file = Path(__file__).parent / "tickers.json"
    with open(tickers_file) as f:
        tickers = [t["ticker"] for t in json.load(f)["watchlist"]]

    conn = sqlite3.connect(DB_PATH)
    total = 0
    for ticker in tickers:
        total += scrape_ticker(ticker, conn)
        time.sleep(1)  # rate limit
    conn.close()
    log.info("Done. Total: %d posts", total)

if __name__ == "__main__":
    run()