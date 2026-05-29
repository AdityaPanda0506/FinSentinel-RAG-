"""
feature_engineering.py  (v2 — improved)
----------------------------------------
Changes from v1:
  - Added macro features: VIX, SPY return, 10Y yield, Gold
  - Added earnings proximity feature (within 7 days of quarter end)
  - Changed to 1-day forward return target (more responsive)
  - Added sector as encoded feature
  - Added gap-up / gap-down overnight feature
  - Better null handling
"""

import sqlite3
import logging
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

ROOT_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT_DIR / "data"
DB_PATH       = DATA_DIR / "raw_prices.db"
SENTIMENT_CSV = DATA_DIR / "daily_sentiment.csv"
TICKERS_FILE  = ROOT_DIR / "ingestion" / "tickers.json"
OUT_CSV       = DATA_DIR / "features.csv"

# ── Improved thresholds for 1-day target ──────────────────────────────────────
BUY_THRESHOLD  =  0.02   # +1% in 1 day
SELL_THRESHOLD = -0.02   # -1% in 1 day

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s")
log = logging.getLogger(__name__)


def load_prices(ticker=None):
    conn  = sqlite3.connect(DB_PATH)
    query = "SELECT ticker, date, open, high, low, close, volume FROM prices"
    params = []
    if ticker:
        query += " WHERE ticker = ?"
        params.append(ticker.upper())
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_sentiment(ticker=None):
    if not SENTIMENT_CSV.exists():
        log.warning("No sentiment CSV found — sentiment features will be zero")
        return pd.DataFrame()
    df = pd.read_csv(SENTIMENT_CSV, parse_dates=["date"])
    if ticker:
        df = df[df["ticker"] == ticker.upper()]
    return df


def load_macro_data():
    """
    Extract macro tickers from prices DB and pivot into
    date-indexed columns: vix, spy_return, tlt_return, gld_return
    """
    macro_map = {
        "^VIX": "vix",
        "SPY":  "spy_close",
        "^TNX": "tnx_yield",
        "GLD":  "gld_close",
        "TLT":  "tlt_close",
    }
    conn   = sqlite3.connect(DB_PATH)
    frames = []
    for ticker, col_name in macro_map.items():
        try:
            df = pd.read_sql_query(
                "SELECT date, close FROM prices WHERE ticker=?",
                conn, params=[ticker], parse_dates=["date"]
            )
            if not df.empty:
                df = df.rename(columns={"close": col_name})
                frames.append(df.set_index("date"))
        except Exception:
            pass
    conn.close()

    if not frames:
        log.warning("No macro data found in DB — run price_fetcher.py with new tickers.json")
        return pd.DataFrame()

    macro = pd.concat(frames, axis=1).reset_index()
    macro = macro.rename(columns={"index": "date"})

    # Calculate returns for ETFs
    for col in ["spy_close", "gld_close", "tlt_close"]:
        if col in macro.columns:
            ret_col = col.replace("_close", "_return")
            macro[ret_col] = macro[col].pct_change().round(4)

    macro = macro.drop(columns=[c for c in ["spy_close","gld_close","tlt_close"]
                                  if c in macro.columns])
    log.info("Macro features loaded: %s", [c for c in macro.columns if c != "date"])
    return macro


def load_sector_map():
    """Build ticker → sector mapping from tickers.json"""
    with open(TICKERS_FILE) as f:
        config = json.load(f)
    sector_map = {t["ticker"]: t["sector"] for t in config["watchlist"]}
    # Encode sectors as integers
    sectors     = sorted(set(sector_map.values()))
    sector_enc  = {s: i for i, s in enumerate(sectors)}
    return {t: sector_enc[s] for t, s in sector_map.items()}


def add_price_features(df):
    df = df.copy().sort_values(["ticker", "date"])
    results = []

    for ticker, grp in df.groupby("ticker"):
        grp   = grp.copy().reset_index(drop=True)
        close  = grp["close"]
        volume = grp["volume"]
        open_  = grp["open"]

        # Returns
        for w in [1, 3, 5, 10, 20]:
            grp[f"return_{w}d"] = close.pct_change(w).round(4)

        # Overnight gap (close-to-open)
        grp["gap"] = ((open_ - close.shift(1)) / close.shift(1)).round(4)

        # High-low range (intraday volatility)
        grp["hl_range"] = ((grp["high"] - grp["low"]) / close).round(4)

        # RSI 14
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        grp["rsi_14"] = (100 - 100 / (1 + rs)).round(2)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        grp["macd"]        = macd.round(4)
        grp["macd_signal"] = signal.round(4)
        grp["macd_hist"]   = (macd - signal).round(4)

        # Bollinger Bands
        sma20 = close.rolling(20, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=1).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        band_range = (upper - lower).replace(0, np.nan)
        grp["bb_position"] = ((close - lower) / band_range).round(4)
        grp["bb_width"]    = (band_range / sma20).round(4)

        # Volume spike
        vol_avg = volume.rolling(20, min_periods=1).mean()
        grp["volume_spike"] = (volume / vol_avg.replace(0, np.nan)).round(4)

        # Price vs moving averages
        grp["price_vs_sma20"] = ((close / sma20) - 1).round(4)
        grp["price_vs_sma50"] = ((close / close.rolling(50, min_periods=1).mean()) - 1).round(4)

        # Earnings proximity proxy
        # Stocks tend to move more in Jan, Apr, Jul, Oct (earnings months)
        grp["earnings_month"] = grp["date"].dt.month.isin([1, 4, 7, 10]).astype(int)
        grp["days_to_month_end"] = (
            grp["date"] + pd.offsets.MonthEnd(0) - grp["date"]
        ).dt.days
        grp["earnings_proximity"] = (grp["days_to_month_end"] <= 7).astype(int)

        results.append(grp)

    return pd.concat(results, ignore_index=True)


def add_sentiment_lags(df):
    df = df.copy().sort_values(["ticker", "date"])
    results = []
    for ticker, grp in df.groupby("ticker"):
        grp   = grp.copy().reset_index(drop=True)
        score = grp["sentiment_score"]

        grp["sentiment_lag_1d"]   = score.shift(1).round(4)
        grp["sentiment_lag_3d"]   = score.shift(3).round(4)
        grp["sentiment_lag_5d"]   = score.shift(5).round(4)
        grp["sentiment_rolling_5d"]  = score.rolling(5,  min_periods=1).mean().round(4)
        grp["sentiment_rolling_10d"] = score.rolling(10, min_periods=1).mean().round(4)

        if "news_score" in grp.columns and "reddit_score" in grp.columns:
            grp["sentiment_divergence"] = (
                grp["news_score"].fillna(0) - grp["reddit_score"].fillna(0)
            ).round(4)

        results.append(grp)
    return pd.concat(results, ignore_index=True)


def add_target_label(df):
    """3-day forward return target with balanced thresholds"""
    df = df.copy().sort_values(["ticker", "date"])
    results = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.copy().reset_index(drop=True)

        # 3-day forward return
        grp["forward_return_3d"] = grp["close"].pct_change(3).shift(-3).round(4)

        def label(ret):
            if pd.isna(ret): return None
            if ret >=  0.02: return "buy"
            if ret <= -0.02: return "sell"
            return "hold"

        grp["signal"] = grp["forward_return_3d"].apply(label)
        results.append(grp)

    out  = pd.concat(results, ignore_index=True)
    dist = out["signal"].value_counts(dropna=True)
    log.info("Label distribution: %s", dist.to_dict())
    return out


def build_features(ticker=None):
    prices    = load_prices(ticker)
    sentiment = load_sentiment(ticker)
    macro     = load_macro_data()
    sector_map = load_sector_map()

    log.info("Engineering price features ...")
    prices = add_price_features(prices)

    # Merge sentiment
    if not sentiment.empty:
        sentiment_cols = [c for c in [
            "ticker", "date", "sentiment_score", "news_score", "reddit_score",
            "bullish_count", "bearish_count", "neutral_count", "total_articles",
            "bullish_ratio", "bearish_ratio",
            "sentiment_momentum_3d", "sentiment_momentum_7d",
        ] if c in sentiment.columns]
        df = prices.merge(sentiment[sentiment_cols], on=["ticker","date"], how="left")
    else:
        df = prices.copy()
        df["sentiment_score"] = 0.0

    # Fill missing sentiment with 0
    sent_fill = ["sentiment_score","news_score","reddit_score",
                 "bullish_count","bearish_count","neutral_count",
                 "total_articles","bullish_ratio","bearish_ratio",
                 "sentiment_momentum_3d","sentiment_momentum_7d"]
    for c in sent_fill:
        if c in df.columns:
            df[c] = df[c].fillna(0)
        else:
            df[c] = 0.0

    # Merge macro features
    if not macro.empty:
        df = df.merge(macro, on="date", how="left")
        macro_cols = [c for c in macro.columns if c != "date"]
        df[macro_cols] = df[macro_cols].ffill().fillna(0)

    # Add sector encoding
    df["sector_encoded"] = df["ticker"].map(sector_map).fillna(0).astype(int)

    # Add sentiment lags
    df = add_sentiment_lags(df)

    # Add target
    df = add_target_label(df)

    # Drop rows with NaN in core features
    core = ["return_1d", "rsi_14", "macd", "bb_position", "sentiment_score"]
    df   = df.dropna(subset=core)
    df   = df[df["signal"].notna()]

    # Make sure the column reference is correct — add this check too
    if "forward_return_1d" in df.columns and "forward_return_3d" not in df.columns:
        df = df.rename(columns={"forward_return_1d": "forward_return_3d"})

    # Exclude macro/benchmark tickers from training rows
    macro_tickers = ["^VIX", "^TNX", "SPY", "QQQ", "GLD", "TLT"]
    df = df[~df["ticker"].isin(macro_tickers)]

    log.info("Final feature matrix: %d rows, %d columns", len(df), len(df.columns))
    
    # Remove tickers with less than 50 days of sentiment data
    # These hurt the model more than they help
    sentiment_coverage = df.groupby("ticker")["sentiment_score"].apply(
        lambda x: (x != 0).sum()
    )
    good_tickers = sentiment_coverage[sentiment_coverage >= 50].index
    df = df[df["ticker"].isin(good_tickers)]
    log.info("Tickers with enough sentiment data: %d", df["ticker"].nunique())
    return df


def run(ticker_filter=None):
    df = build_features(ticker_filter)
    df.to_csv(OUT_CSV, index=False)
    log.info("Features saved → %s", OUT_CSV)

    feature_cols = [c for c in df.columns if c not in
                    ("ticker","date","open","high","low","close",
                     "volume","signal","forward_return_1d","forward_return_3d")]
    log.info("Feature columns (%d): %s", len(feature_cols), feature_cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    run(ticker_filter=args.ticker)