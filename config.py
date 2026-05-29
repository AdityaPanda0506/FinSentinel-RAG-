"""
config.py
---------
Central config file for FinSentinel.
All modules import constants from here — never hardcode values in scripts.

Loads secrets from .env automatically via python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

# ── API Keys ───────────────────────────────────────────────────────────────────
NEWSAPI_KEY          = os.getenv("NEWSAPI_KEY",          "")
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT",    "finsentinel_v1")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Ticker watchlist path ──────────────────────────────────────────────────────
TICKERS_FILE = ROOT_DIR / "ingestion" / "tickers.json"

# ── Model settings ─────────────────────────────────────────────────────────────
FINBERT_MODEL   = "ProsusAI/finbert"
BATCH_SIZE      = 32

# ── Strategy settings ──────────────────────────────────────────────────────────
BUY_THRESHOLD   =  0.02     # +2% forward return → BUY label
SELL_THRESHOLD  = -0.02     # -2% forward return → SELL label
HOLD_DAYS       = 3         # hold each position N trading days
RISK_FREE_RATE  = 0.05      # annual, for Sharpe calculation

# ── Scraper settings ───────────────────────────────────────────────────────────
NEWS_DAYS_BACK  = 28        # max lookback for NewsAPI free tier
REDDIT_POSTS    = 100       # posts per subreddit per run
SUBREDDITS      = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
]

# ── Validation ─────────────────────────────────────────────────────────────────
def validate() -> list[str]:
    """
    Check all required config values are set.
    Returns list of warning messages (empty = all good).
    """
    warnings = []
    if not NEWSAPI_KEY:
        warnings.append("NEWSAPI_KEY not set — news scraping will fail")
    if not REDDIT_CLIENT_ID:
        warnings.append("REDDIT_CLIENT_ID not set — Reddit scraping will fail")
    if not REDDIT_CLIENT_SECRET:
        warnings.append("REDDIT_CLIENT_SECRET not set — Reddit scraping will fail")
    return warnings


if __name__ == "__main__":
    issues = validate()
    if issues:
        for w in issues:
            print(f"⚠  {w}")
    else:
        print("✓  All config values loaded successfully")