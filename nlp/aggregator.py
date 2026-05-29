"""
aggregator.py
-------------
Takes raw per-headline sentiment scores from sentiment_scores.csv
and aggregates them into a single daily sentiment signal per ticker.

Why aggregate?
    FinBERT scores individual headlines. But your ML model needs
    ONE number per ticker per day — not 50 scattered scores.
    This file collapses them into meaningful daily features.

Aggregation strategy:
    - Volume-weighted average sentiment (more articles = stronger signal)
    - Separate news vs Reddit sentiment (different market segments)
    - Sentiment momentum (how fast is sentiment changing?)
    - Bullish/bearish article count ratio

Usage:
    python nlp/aggregator.py              # aggregate all tickers
    python nlp/aggregator.py --ticker AAPL

Output:
    data/daily_sentiment.csv
    columns: ticker, date, sentiment_score, news_score, reddit_score,
             bullish_count, bearish_count, neutral_count, total_articles,
             sentiment_momentum_3d, sentiment_momentum_7d
"""

import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT_DIR / "data"
SCORES_CSV    = DATA_DIR / "sentiment_scores.csv"
OUT_CSV       = DATA_DIR / "daily_sentiment.csv"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────
def net_score(row: pd.Series) -> float:
    """
    Net sentiment = positive_prob - negative_prob
    Range: [-1, +1]
        +1 = fully bullish
        -1 = fully bearish
         0 = neutral
    This is more informative than just the winning label.
    """
    return row["positive"] - row["negative"]


def confidence_weighted_mean(group: pd.DataFrame) -> float:
    """
    Weighted average of net_score, where the weight is confidence.
    High-confidence predictions count more than low-confidence ones.
    """
    weights = group["confidence"]
    scores  = group["net_score"]
    if weights.sum() == 0:
        return 0.0
    return float(np.average(scores, weights=weights))


# ── Core aggregation ───────────────────────────────────────────────────────────
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-headline scores into one row per (ticker, date).

    Returns a DataFrame with:
        ticker, date,
        sentiment_score       — overall confidence-weighted net sentiment
        news_score            — same, but only from news source
        reddit_score          — same, but only from reddit source
        bullish_count         — number of bullish articles that day
        bearish_count         — number of bearish articles that day
        neutral_count         — number of neutral articles that day
        total_articles        — total articles scored that day
        bullish_ratio         — bullish / total (useful ML feature)
        bearish_ratio         — bearish / total
    """
    df = df.copy()
    df["net_score"] = df.apply(net_score, axis=1)

    results = []

    for (ticker, date), group in df.groupby(["ticker", "published_at"]):
        # Overall sentiment (all sources)
        overall = confidence_weighted_mean(group)

        # Source-split sentiment
        news_grp   = group[group["source"] == "news"]
        reddit_grp = group[group["source"] == "reddit"]

        news_score   = confidence_weighted_mean(news_grp)   if not news_grp.empty   else np.nan
        reddit_score = confidence_weighted_mean(reddit_grp) if not reddit_grp.empty else np.nan

        # Count breakdown
        label_counts = group["label"].value_counts()
        bullish  = int(label_counts.get("positive", 0))
        bearish  = int(label_counts.get("negative", 0))
        neutral  = int(label_counts.get("neutral",  0))
        total    = len(group)

        results.append({
            "ticker":          ticker,
            "date":            date,
            "sentiment_score": round(overall, 4),
            "news_score":      round(news_score, 4)   if not np.isnan(news_score)   else None,
            "reddit_score":    round(reddit_score, 4) if not np.isnan(reddit_score) else None,
            "bullish_count":   bullish,
            "bearish_count":   bearish,
            "neutral_count":   neutral,
            "total_articles":  total,
            "bullish_ratio":   round(bullish / total, 4) if total > 0 else 0.0,
            "bearish_ratio":   round(bearish / total, 4) if total > 0 else 0.0,
        })

    out = pd.DataFrame(results)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling sentiment momentum features per ticker.

    sentiment_momentum_3d = today's score - 3-day rolling average
    sentiment_momentum_7d = today's score - 7-day rolling average

    Positive momentum = sentiment improving (potential buy signal)
    Negative momentum = sentiment deteriorating (potential sell signal)
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"])

    for ticker, group in df.groupby("ticker"):
        idx   = group.index
        score = group["sentiment_score"]

        roll3 = score.rolling(window=3, min_periods=1).mean()
        roll7 = score.rolling(window=7, min_periods=1).mean()

        df.loc[idx, "sentiment_momentum_3d"] = (score - roll3).round(4)
        df.loc[idx, "sentiment_momentum_7d"] = (score - roll7).round(4)

    return df


def add_sentiment_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each day into a sentiment regime — useful for
    the dashboard and as a categorical ML feature.

    Regimes:
        strongly_bullish  → score > +0.3
        mildly_bullish    → score  0.1 to 0.3
        neutral           → score -0.1 to 0.1
        mildly_bearish    → score -0.3 to -0.1
        strongly_bearish  → score < -0.3
    """
    def classify(score: float) -> str:
        if score > 0.3:
            return "strongly_bullish"
        elif score > 0.1:
            return "mildly_bullish"
        elif score > -0.1:
            return "neutral"
        elif score > -0.3:
            return "mildly_bearish"
        else:
            return "strongly_bearish"

    df["sentiment_regime"] = df["sentiment_score"].apply(classify)
    return df


# ── Quality checks ─────────────────────────────────────────────────────────────
def run_quality_checks(df: pd.DataFrame) -> None:
    """
    Log basic quality stats — useful to catch data issues early.
    """
    log.info("─" * 60)
    log.info("Quality checks:")
    log.info("  Total rows      : %d", len(df))
    log.info("  Tickers covered : %d", df["ticker"].nunique())
    log.info("  Date range      : %s → %s",
             df["date"].min().date(), df["date"].max().date())

    # Check for tickers with very few data points
    counts = df.groupby("ticker").size()
    sparse = counts[counts < 10]
    if not sparse.empty:
        log.warning("  Tickers with <10 daily rows (may affect ML quality): %s",
                    ", ".join(sparse.index.tolist()))

    # Overall sentiment distribution
    regime_dist = df["sentiment_regime"].value_counts(normalize=True)
    log.info("  Sentiment regime distribution:")
    for regime, pct in regime_dist.items():
        log.info("    %-20s  %.1f%%", regime, pct * 100)

    # Top bullish / bearish tickers overall
    avg = df.groupby("ticker")["sentiment_score"].mean().sort_values()
    log.info("  Most bearish tickers: %s",
             ", ".join(f"{t}({s:.2f})" for t, s in avg.head(3).items()))
    log.info("  Most bullish tickers: %s",
             ", ".join(f"{t}({s:.2f})" for t, s in avg.tail(3).items()))


# ── Main ───────────────────────────────────────────────────────────────────────
def run(ticker_filter: str | None = None) -> pd.DataFrame:
    if not SCORES_CSV.exists():
        log.error("sentiment_scores.csv not found at %s. Run sentiment_pipeline.py first.", SCORES_CSV)
        return pd.DataFrame()

    log.info("Loading sentiment scores from %s ...", SCORES_CSV)
    df = pd.read_csv(SCORES_CSV)

    required_cols = {"ticker", "published_at", "source", "label",
                     "positive", "negative", "neutral", "confidence"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error("Missing columns in sentiment_scores.csv: %s", missing)
        return pd.DataFrame()

    if ticker_filter:
        df = df[df["ticker"] == ticker_filter.upper()]
        if df.empty:
            log.warning("No data for ticker %s", ticker_filter)
            return pd.DataFrame()

    log.info("Aggregating %d scored headlines ...", len(df))

    daily = aggregate_daily(df)
    daily = add_momentum_features(daily)
    daily = add_sentiment_regime(daily)

    # Save
    daily.to_csv(OUT_CSV, index=False)
    log.info("Daily sentiment saved → %s  (%d rows)", OUT_CSV, len(daily))

    run_quality_checks(daily)
    return daily


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate sentiment scores to daily level")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Aggregate only for one ticker (e.g. --ticker AAPL)")
    args = parser.parse_args()
    run(ticker_filter=args.ticker)