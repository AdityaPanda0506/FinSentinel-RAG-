"""
entity_extractor.py
--------------------
Extracts stock ticker symbols from raw text using two strategies:
    1. Regex — catches $AAPL style mentions and ALL-CAPS words
    2. spaCy NER — catches company names like "Apple" and maps to ticker

This is used to:
    - Improve ticker tagging on news articles that mention
      "Apple" instead of "$AAPL"
    - Re-tag Reddit posts that weren't caught by the simple
      regex in reddit_scraper.py
    - Optionally, re-process any existing DB rows with null tickers

spaCy setup (run once):
    pip install spacy
    python -m spacy download en_core_web_sm

Usage:
    # Extract tickers from a single text string
    from nlp.entity_extractor import extract_tickers
    tickers = extract_tickers("Apple beat earnings expectations today")
    # → ['AAPL']

    # Re-tag all news articles missing a ticker
    python nlp/entity_extractor.py --retag-news

    # Re-tag all Reddit posts
    python nlp/entity_extractor.py --retag-reddit
"""

import re
import sqlite3
import json
import logging
import argparse
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
DB_PATH      = DATA_DIR / "raw_news.db"
TICKERS_FILE = Path(__file__).parent.parent / "ingestion" / "tickers.json"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Ticker + company name lookup table ─────────────────────────────────────────
def build_lookup() -> tuple[set[str], dict[str, str]]:
    """
    Returns:
        known_tickers   — set of ticker strings e.g. {"AAPL", "MSFT", ...}
        name_to_ticker  — dict mapping company name variations to ticker
                          e.g. {"apple": "AAPL", "microsoft": "MSFT", ...}
    """
    with open(TICKERS_FILE) as f:
        config = json.load(f)

    known_tickers  = set()
    name_to_ticker = {}

    for entry in config["watchlist"]:
        ticker = entry["ticker"]
        name   = entry["name"]

        known_tickers.add(ticker)

        # Map full company name (lowercased) → ticker
        name_to_ticker[name.lower()] = ticker

        # Also map first word of company name (e.g. "apple" → AAPL)
        first_word = name.split()[0].lower()
        if len(first_word) > 3:  # skip short words like "The", "US"
            name_to_ticker[first_word] = ticker

    return known_tickers, name_to_ticker


KNOWN_TICKERS, NAME_TO_TICKER = build_lookup()

# Common false positives to ignore
TICKER_BLACKLIST = {
    "I", "A", "AT", "IT", "ON", "BE", "BY", "OR", "US", "UK",
    "GDP", "CEO", "CFO", "IPO", "ETF", "FOR", "NOW", "ALL",
    "ARE", "GET", "NEW", "TOP", "DUE", "CUT", "BUY", "NOT",
}


# ── Regex-based extraction ─────────────────────────────────────────────────────
def extract_by_regex(text: str) -> set[str]:
    """
    Strategy 1: Pattern matching.
    Finds:
        - $TICKER mentions  (e.g. $AAPL, $TSLA)
        - ALL-CAPS 2-5 letter words matching known tickers
    """
    found = set()
    upper_text = text.upper()

    # $TICKER pattern (most reliable)
    for match in re.finditer(r"\$([A-Z]{1,5})\b", upper_text):
        t = match.group(1)
        if t in KNOWN_TICKERS:
            found.add(t)

    # ALL-CAPS ticker-like words
    for match in re.finditer(r"\b([A-Z]{2,5})\b", upper_text):
        t = match.group(1)
        if t in KNOWN_TICKERS and t not in TICKER_BLACKLIST:
            found.add(t)

    return found


# ── spaCy NER-based extraction ─────────────────────────────────────────────────
_nlp = None  # lazy load — only imports spaCy when actually called


def get_spacy():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            log.info("spaCy model loaded (en_core_web_sm)")
        except OSError:
            log.warning(
                "spaCy model 'en_core_web_sm' not found.\n"
                "Run: python -m spacy download en_core_web_sm\n"
                "Falling back to regex-only extraction."
            )
            _nlp = None
    return _nlp


def extract_by_ner(text: str) -> set[str]:
    """
    Strategy 2: Named Entity Recognition.
    Finds ORG entities (e.g. "Apple Inc", "Tesla") and
    maps them to tickers using our name_to_ticker lookup.
    """
    nlp = get_spacy()
    if nlp is None:
        return set()

    doc   = nlp(text)
    found = set()

    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT"):
            name_lower = ent.text.lower().strip()

            # Direct match
            if name_lower in NAME_TO_TICKER:
                found.add(NAME_TO_TICKER[name_lower])
                continue

            # Partial match — check if any known name is a substring
            for known_name, ticker in NAME_TO_TICKER.items():
                if known_name in name_lower or name_lower in known_name:
                    found.add(ticker)
                    break

    return found


# ── Combined extraction (public API) ───────────────────────────────────────────
def extract_tickers(text: str, use_ner: bool = True) -> list[str]:
    """
    Main function — combines regex + NER for best recall.

    Args:
        text     : raw text string (headline, post title, body)
        use_ner  : whether to run spaCy NER (slower but catches
                   company names like "Apple" without $)

    Returns:
        Sorted list of unique ticker strings e.g. ['AAPL', 'MSFT']

    Example:
        >>> extract_tickers("Apple and Microsoft both beat earnings today")
        ['AAPL', 'MSFT']
        >>> extract_tickers("$TSLA up 5% after delivery numbers")
        ['TSLA']
    """
    if not text or not isinstance(text, str):
        return []

    found = extract_by_regex(text)

    if use_ner:
        found |= extract_by_ner(text)

    return sorted(found)


# ── Re-tagger: update DB rows missing tickers ──────────────────────────────────
def retag_news(conn: sqlite3.Connection) -> int:
    """
    Find news articles with no ticker assigned and try to extract one.
    Updates the DB in-place.
    Returns number of rows updated.
    """
    log.info("Re-tagging news articles with missing tickers ...")

    # news table stores one ticker per row — find rows with generic/missing ticker
    rows = conn.execute("""
        SELECT id, headline FROM news
        WHERE ticker IS NULL OR ticker = '' OR ticker = 'UNKNOWN'
    """).fetchall()

    log.info("  Found %d articles to re-tag", len(rows))

    updated = 0
    for row_id, headline in rows:
        tickers = extract_tickers(headline or "")
        if tickers:
            # Update each ticker as a separate row
            # (first ticker is the primary; others get duplicate rows)
            conn.execute(
                "UPDATE news SET ticker = ? WHERE id = ?",
                (tickers[0], row_id)
            )
            updated += 1

            # Insert extra rows for additional tickers
            for extra_ticker in tickers[1:]:
                conn.execute("""
                    INSERT OR IGNORE INTO news (ticker, headline, description,
                        source, url, published_at, fetched_at)
                    SELECT ?, headline, description, source, url, published_at, fetched_at
                    FROM news WHERE id = ?
                """, (extra_ticker, row_id))

    conn.commit()
    log.info("  Updated %d rows", updated)
    return updated


def retag_reddit(conn: sqlite3.Connection) -> int:
    """
    Re-run ticker extraction on Reddit posts where tickers is NULL.
    Updates the DB in-place.
    """
    log.info("Re-tagging Reddit posts with missing tickers ...")

    rows = conn.execute("""
        SELECT id, title, selftext FROM reddit_posts
        WHERE tickers IS NULL OR tickers = ''
    """).fetchall()

    log.info("  Found %d posts to re-tag", len(rows))

    updated = 0
    for row_id, title, selftext in rows:
        full_text = f"{title or ''} {selftext or ''}"
        tickers   = extract_tickers(full_text)
        if tickers:
            conn.execute(
                "UPDATE reddit_posts SET tickers = ? WHERE id = ?",
                (",".join(tickers), row_id)
            )
            updated += 1

    conn.commit()
    log.info("  Updated %d posts", updated)
    return updated


# ── Quick test ─────────────────────────────────────────────────────────────────
def run_demo() -> None:
    """
    Quick sanity check — run extraction on a few test sentences
    and print the results. Good for verifying your setup works.
    """
    test_cases = [
        "$AAPL reports record earnings, stock up 8%",
        "Apple and Microsoft both beat Q4 expectations",
        "Tesla faces recall issues in China, $TSLA drops",
        "The Federal Reserve raised rates by 25 basis points",
        "NVDA hits all-time high as AI demand surges",
        "JPMorgan upgrades Goldman Sachs to overweight",
        "General market outlook remains uncertain in 2024",
    ]

    log.info("Running entity extraction demo:")
    log.info("─" * 60)

    for text in test_cases:
        tickers = extract_tickers(text)
        log.info("TEXT   : %s", text)
        log.info("TICKERS: %s", tickers if tickers else "(none detected)")
        log.info("")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and retag ticker symbols in the database"
    )
    parser.add_argument("--retag-news",   action="store_true",
                        help="Re-tag news articles with missing tickers")
    parser.add_argument("--retag-reddit", action="store_true",
                        help="Re-tag Reddit posts with missing tickers")
    parser.add_argument("--demo",         action="store_true",
                        help="Run a quick demo on test sentences")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.retag_news or args.retag_reddit:
        conn = sqlite3.connect(DB_PATH)
        if args.retag_news:
            retag_news(conn)
        if args.retag_reddit:
            retag_reddit(conn)
        conn.close()
    else:
        parser.print_help()