"""
reddit_scraper.py
-----------------
Scrapes posts and top comments from finance subreddits
using PRAW (Python Reddit API Wrapper) and stores them
in a local SQLite database, tagged by ticker where detectable.

Requires:
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT in .env
    pip install praw python-dotenv

Setup (takes ~5 minutes):
    1. Go to https://www.reddit.com/prefs/apps
    2. Click "Create another app" → choose "script"
    3. Name it "finsentinel", set redirect to http://localhost:8080
    4. Copy the client_id (under app name) and client_secret
    5. Add to .env:
        REDDIT_CLIENT_ID=your_id_here
        REDDIT_CLIENT_SECRET=your_secret_here
        REDDIT_USER_AGENT=finsentinel_scraper_v1

Usage:
    python reddit_scraper.py              # scrape all configured subreddits
    python reddit_scraper.py --sub wallstreetbets

Output:
    data/raw_news.db  →  table: reddit_posts (same DB as news)
"""

import sqlite3
import re
import os
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import praw
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH  = DATA_DIR / "raw_news.db"

DATA_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "finsentinel_scraper_v1")

# Subreddits to monitor
SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
]

# How many posts to pull per subreddit per run
POSTS_LIMIT = 100

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Database setup ─────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection) -> None:
    """Create reddit_posts table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reddit_posts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id      TEXT    NOT NULL UNIQUE,
            subreddit    TEXT    NOT NULL,
            title        TEXT    NOT NULL,
            selftext     TEXT,
            score        INTEGER,
            num_comments INTEGER,
            tickers      TEXT,             -- comma-separated tickers detected
            url          TEXT,
            created_utc  TEXT,
            fetched_at   TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_reddit_tickers
        ON reddit_posts(tickers)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_reddit_date
        ON reddit_posts(created_utc)
    """)
    conn.commit()
    log.info("reddit_posts table ready → %s", DB_PATH)


# ── Ticker extraction ──────────────────────────────────────────────────────────
# Load known tickers from tickers.json for accurate matching
def load_known_tickers() -> set[str]:
    tickers_file = Path(__file__).parent / "tickers.json"
    try:
        import json
        with open(tickers_file) as f:
            data = json.load(f)
        return {t["ticker"] for t in data["watchlist"]}
    except Exception:
        # Fallback to a small hardcoded set if file not found
        return {
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META",
            "JPM", "BAC", "GS", "JNJ", "PFE", "XOM", "WMT", "DIS",
            "NFLX", "AMD", "INTC", "V", "MA", "PYPL", "CRM", "BA",
        }

KNOWN_TICKERS = load_known_tickers()

# Common false-positive single-letter/word tickers to exclude
TICKER_BLACKLIST = {
    "I", "A", "AT", "IT", "ON", "BE", "BY", "OR", "US", "UK",
    "GDP", "CEO", "CFO", "IPO", "ETF", "SPY", "QQQ", "DD",
}

def extract_tickers(text: str) -> list[str]:
    """
    Detect stock ticker mentions in text using regex.
    Looks for:
      - $AAPL style (explicit)
      - ALL-CAPS 2-5 letter words that match our known tickers

    Returns a deduplicated list of matched tickers.
    """
    found = set()

    # $TICKER mentions (most reliable)
    dollar_mentions = re.findall(r"\$([A-Z]{1,5})\b", text.upper())
    for t in dollar_mentions:
        if t in KNOWN_TICKERS:
            found.add(t)

    # ALL-CAPS words that match known tickers
    caps_words = re.findall(r"\b([A-Z]{2,5})\b", text.upper())
    for t in caps_words:
        if t in KNOWN_TICKERS and t not in TICKER_BLACKLIST:
            found.add(t)

    return sorted(found)


# ── Reddit client ──────────────────────────────────────────────────────────────
def get_reddit_client() -> praw.Reddit:
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        raise EnvironmentError(
            "Reddit credentials not found.\n"
            "Add to .env:\n"
            "  REDDIT_CLIENT_ID=...\n"
            "  REDDIT_CLIENT_SECRET=...\n"
            "  REDDIT_USER_AGENT=finsentinel_scraper_v1\n"
            "Get credentials at: https://www.reddit.com/prefs/apps"
        )
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        ratelimit_seconds=5,  # auto-respects Reddit's rate limits
    )


# ── Scraping ───────────────────────────────────────────────────────────────────
def scrape_subreddit(
    reddit: praw.Reddit,
    subreddit_name: str,
    limit: int = POSTS_LIMIT,
) -> list[dict]:
    """
    Pull the top `limit` hot posts from a subreddit.
    Also pulls the top 3 comments per post for richer signal.
    """
    log.info("Scraping r/%s  (limit=%d)", subreddit_name, limit)
    posts = []

    try:
        subreddit = reddit.subreddit(subreddit_name)
        # Pull hot + new posts for fresh signal
        submissions = list(subreddit.hot(limit=limit // 2)) + \
                      list(subreddit.new(limit=limit // 2))
    except Exception as exc:
        log.error("Failed to access r/%s: %s", subreddit_name, exc)
        return []

    fetched_at = datetime.now(timezone.utc).isoformat()

    for submission in submissions:
        title    = submission.title or ""
        selftext = (submission.selftext or "")[:1000]  # cap at 1000 chars
        full_text = f"{title} {selftext}"

        tickers = extract_tickers(full_text)

        # Pull top 3 comments for extra ticker signal
        try:
            submission.comments.replace_more(limit=0)
            top_comments = submission.comments.list()[:3]
            comments_text = " ".join(
                c.body for c in top_comments if hasattr(c, "body")
            )
            tickers += extract_tickers(comments_text)
            tickers = sorted(set(tickers))
        except Exception:
            pass  # comments are a bonus, not required

        # Convert Reddit epoch timestamp to ISO date
        created_utc = datetime.fromtimestamp(
            submission.created_utc, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        posts.append({
            "post_id":      submission.id,
            "subreddit":    subreddit_name,
            "title":        title.strip(),
            "selftext":     selftext.strip(),
            "score":        submission.score,
            "num_comments": submission.num_comments,
            "tickers":      ",".join(tickers) if tickers else None,
            "url":          f"https://reddit.com{submission.permalink}",
            "created_utc":  created_utc,
            "fetched_at":   fetched_at,
        })

    log.info("  → %d posts scraped from r/%s", len(posts), subreddit_name)
    return posts


def save_posts(posts: list[dict], conn: sqlite3.Connection) -> int:
    """Insert posts into DB. Skips duplicates via UNIQUE(post_id)."""
    if not posts:
        return 0

    cursor   = conn.cursor()
    inserted = 0

    for post in posts:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO reddit_posts
                    (post_id, subreddit, title, selftext, score,
                     num_comments, tickers, url, created_utc, fetched_at)
                VALUES
                    (:post_id, :subreddit, :title, :selftext, :score,
                     :num_comments, :tickers, :url, :created_utc, :fetched_at)
            """, post)
            inserted += cursor.rowcount
        except sqlite3.Error as exc:
            log.warning("DB insert error: %s", exc)

    conn.commit()
    return inserted


# ── Quick stats ────────────────────────────────────────────────────────────────
def print_stats(conn: sqlite3.Connection) -> None:
    """Print a quick summary of what's in the DB."""
    total = conn.execute("SELECT COUNT(*) FROM reddit_posts").fetchone()[0]
    with_tickers = conn.execute(
        "SELECT COUNT(*) FROM reddit_posts WHERE tickers IS NOT NULL"
    ).fetchone()[0]
    log.info("reddit_posts: %d total, %d with detected tickers", total, with_tickers)

    top = conn.execute("""
        SELECT subreddit, COUNT(*) as n
        FROM reddit_posts
        GROUP BY subreddit
        ORDER BY n DESC
    """).fetchall()
    for row in top:
        log.info("  r/%-20s  %d posts", row[0], row[1])


# ── Main orchestrator ──────────────────────────────────────────────────────────
def run(sub_filter: str | None = None) -> None:
    subs = SUBREDDITS
    if sub_filter:
        sub_filter = sub_filter.lower().strip()
        subs = [s for s in subs if s.lower() == sub_filter]
        if not subs:
            log.error(
                "Subreddit '%s' not in configured list: %s",
                sub_filter, SUBREDDITS
            )
            return

    reddit = get_reddit_client()
    conn   = sqlite3.connect(DB_PATH)
    init_db(conn)

    total_inserted = 0

    for sub in subs:
        posts    = scrape_subreddit(reddit, sub)
        inserted = save_posts(posts, conn)
        total_inserted += inserted
        log.info("  ✓ r/%s  →  %d new posts saved", sub, inserted)
        time.sleep(2)  # be respectful — Reddit limits ~60 req/min

    print_stats(conn)
    conn.close()

    log.info("─" * 60)
    log.info("Done. Total new posts: %d", total_inserted)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape finance subreddits for sentiment data"
    )
    parser.add_argument(
        "--sub", type=str, default=None,
        help="Scrape a single subreddit (e.g. --sub wallstreetbets)"
    )
    args = parser.parse_args()
    run(sub_filter=args.sub)