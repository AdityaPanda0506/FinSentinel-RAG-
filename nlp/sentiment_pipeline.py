"""
sentiment_pipeline.py
---------------------
Runs FinBERT (finance-tuned BERT) on every headline and Reddit post
in raw_news.db and writes sentiment scores to data/sentiment_scores.csv

FinBERT outputs three probabilities per text:
    positive  →  bullish signal
    negative  →  bearish signal
    neutral   →  no signal

We store all three + the winning label + confidence score.

Model: ProsusAI/finbert  (HuggingFace Hub, ~440MB, auto-downloaded once)

Usage:
    python nlp/sentiment_pipeline.py              # score everything
    python nlp/sentiment_pipeline.py --source news    # only NewsAPI articles
    python nlp/sentiment_pipeline.py --source reddit  # only Reddit posts
    python nlp/sentiment_pipeline.py --ticker AAPL    # one ticker only

Output:
    data/sentiment_scores.csv
    columns: source, ticker, text, published_at,
             label, positive, negative, neutral, confidence
"""

import sqlite3
import logging
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH  = DATA_DIR / "raw_news.db"
OUT_CSV  = DATA_DIR / "sentiment_scores.csv"

# ── Config ─────────────────────────────────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"   # best open-source finance sentiment model
BATCH_SIZE    = 32                   # increase to 64 if you have a GPU
MAX_LENGTH    = 512                  # FinBERT's max token length

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Model loader ───────────────────────────────────────────────────────────────
def load_finbert():
    """
    Load FinBERT tokenizer and model from HuggingFace Hub.
    Downloads ~440MB on first run, then cached locally.
    Automatically uses GPU if available, otherwise CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading FinBERT on %s ...", device.upper())

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model     = model.to(device)
    model.eval()

    log.info("FinBERT loaded. Labels: %s", model.config.id2label)
    return tokenizer, model, device


# ── Inference ──────────────────────────────────────────────────────────────────
def score_batch(
    texts: list[str],
    tokenizer,
    model,
    device: str,
) -> list[dict]:
    """
    Run FinBERT on a batch of texts.
    Returns a list of dicts with keys:
        label, positive, negative, neutral, confidence
    """
    # Tokenize — truncate long texts to MAX_LENGTH
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)

    # Softmax → probabilities
    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

    # FinBERT label order: positive=0, negative=1, neutral=2
    # (verify with model.config.id2label if results look wrong)
    label_map = model.config.id2label  # {0: 'positive', 1: 'negative', 2: 'neutral'}

    results = []
    for row in probs:
        scores = {label_map[i]: float(row[i]) for i in range(len(row))}
        winning_label = max(scores, key=scores.get)
        results.append({
            "label":      winning_label,
            "positive":   round(scores.get("positive", 0.0), 4),
            "negative":   round(scores.get("negative", 0.0), 4),
            "neutral":    round(scores.get("neutral",  0.0), 4),
            "confidence": round(scores[winning_label], 4),
        })

    return results


def score_dataframe(
    df: pd.DataFrame,
    text_col: str,
    tokenizer,
    model,
    device: str,
) -> pd.DataFrame:
    """
    Score all rows in a DataFrame in batches.
    Adds columns: label, positive, negative, neutral, confidence
    """
    texts   = df[text_col].fillna("").tolist()
    all_res = []

    total   = len(texts)
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        res   = score_batch(batch, tokenizer, model, device)
        all_res.extend(res)

        if (i // BATCH_SIZE) % 5 == 0:
            log.info("  Scored %d / %d texts ...", min(i + BATCH_SIZE, total), total)

    scores_df = pd.DataFrame(all_res)
    return pd.concat([df.reset_index(drop=True), scores_df], axis=1)


# ── Data loaders ───────────────────────────────────────────────────────────────
def load_news(conn: sqlite3.Connection, ticker: str | None = None) -> pd.DataFrame:
    """Load news headlines from the DB."""
    query = """
        SELECT
            'news'       AS source,
            ticker,
            headline     AS text,
            published_at
        FROM news
        WHERE headline IS NOT NULL
          AND headline != ''
    """
    params = []
    if ticker:
        query  += " AND ticker = ?"
        params.append(ticker.upper())

    df = pd.read_sql_query(query, conn, params=params)
    log.info("Loaded %d news articles", len(df))
    return df


def load_reddit(conn: sqlite3.Connection, ticker: str | None = None) -> pd.DataFrame:
    """
    Load Reddit posts from the DB.
    Explodes multi-ticker posts so each ticker gets its own row.
    """
    query = """
        SELECT
            'reddit'     AS source,
            tickers,
            title        AS text,
            created_utc  AS published_at
        FROM reddit_posts
        WHERE title IS NOT NULL
          AND title != ''
    """
    df = pd.read_sql_query(query, conn)

    if df.empty:
        log.info("No Reddit posts found in DB")
        return pd.DataFrame()

    # Explode comma-separated tickers into one row per ticker
    df["ticker"] = df["tickers"].str.split(",")
    df = df.explode("ticker").drop(columns=["tickers"])
    df["ticker"] = df["ticker"].str.strip()
    df = df[df["ticker"].notna() & (df["ticker"] != "")]

    if ticker:
        df = df[df["ticker"] == ticker.upper()]

    log.info("Loaded %d Reddit post-ticker pairs", len(df))
    return df[["source", "ticker", "text", "published_at"]]


# ── Output ─────────────────────────────────────────────────────────────────────
def save_scores(df: pd.DataFrame) -> None:
    """
    Append new scores to sentiment_scores.csv.
    De-duplicates by (source, ticker, text) to avoid re-scoring.
    """
    cols = ["source", "ticker", "text", "published_at",
            "label", "positive", "negative", "neutral", "confidence"]
    df = df[cols]

    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["source", "ticker", "text"])
    else:
        combined = df

    combined.to_csv(OUT_CSV, index=False)
    log.info("Saved %d total scored rows → %s", len(combined), OUT_CSV)


def print_summary(df: pd.DataFrame) -> None:
    """Print a quick sentiment breakdown to console."""
    if df.empty:
        return
    log.info("─" * 60)
    log.info("Sentiment breakdown:")
    counts = df["label"].value_counts()
    total  = len(df)
    for label, count in counts.items():
        pct = 100 * count / total
        log.info("  %-10s  %5d  (%.1f%%)", label, count, pct)

    log.info("Per-ticker average sentiment score (positive - negative):")
    df["net_sentiment"] = df["positive"] - df["negative"]
    top = (
        df.groupby("ticker")["net_sentiment"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    for ticker, score in top.items():
        bar   = "▓" * int(abs(score) * 20)
        sign  = "+" if score >= 0 else "-"
        log.info("  %-6s  %s%s  %.3f", ticker, sign, bar, score)


# ── Main ───────────────────────────────────────────────────────────────────────
def run(source_filter: str | None = None, ticker_filter: str | None = None) -> None:
    if not DB_PATH.exists():
        log.error("DB not found at %s. Run ingestion scripts first.", DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)

    frames = []
    if source_filter in (None, "news"):
        frames.append(load_news(conn, ticker_filter))
    if source_filter in (None, "reddit"):
        frames.append(load_reddit(conn, ticker_filter))

    conn.close()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]

    if df.empty:
        log.warning("No text data found. Check your DB.")
        return

    log.info("Total texts to score: %d", len(df))

    tokenizer, model, device = load_finbert()

    log.info("Running FinBERT inference ...")
    df = score_dataframe(df, "text", tokenizer, model, device)

    save_scores(df)
    print_summary(df)
    log.info("Done.")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FinBERT sentiment scoring")
    parser.add_argument("--source", choices=["news", "reddit"], default=None,
                        help="Score only 'news' or 'reddit'. Default: both.")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Score only a specific ticker (e.g. --ticker AAPL)")
    args = parser.parse_args()
    run(source_filter=args.source, ticker_filter=args.ticker)