"""
predict.py
----------
Loads the trained XGBoost model and generates buy / hold / sell
signals for the most recent available data.

Can be called:
    - Standalone (produces a signal report CSV)
    - As a module by dashboard/app.py for live signal display

Usage:
    python ml/predict.py                    # signals for all tickers today
    python ml/predict.py --ticker AAPL      # single ticker
    python ml/predict.py --date 2024-03-15  # signals as of a specific date

Output:
    data/latest_signals.csv
    columns: ticker, date, signal, confidence, probabilities, reasoning
"""

import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ml.train import load_artifacts

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
FEATURES_CSV = DATA_DIR / "features.csv"
OUT_CSV      = DATA_DIR / "latest_signals.csv"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Signal reasoning ───────────────────────────────────────────────────────────
def build_reasoning(row: pd.Series, signal: str) -> str:
    """
    Generate a human-readable explanation for each signal.
    This is what makes the dashboard compelling —
    not just "BUY" but WHY.

    Example output:
        "BUY — Sentiment strongly bullish (+0.62, 3-day momentum +0.18).
         RSI at 42 (not overbought). Price +1.8% over 5 days with
         volume spike 1.8x average."
    """
    parts = []

    # Sentiment reasoning
    sent = row.get("sentiment_score", 0)
    mom  = row.get("sentiment_momentum_3d", 0)
    if abs(sent) > 0.1:
        direction = "bullish" if sent > 0 else "bearish"
        parts.append(
            f"Sentiment {direction} ({sent:+.2f}, "
            f"3d momentum {mom:+.2f})"
        )

    # RSI reasoning
    rsi = row.get("rsi_14", 50)
    if rsi < 35:
        parts.append(f"RSI oversold ({rsi:.0f})")
    elif rsi > 70:
        parts.append(f"RSI overbought ({rsi:.0f})")
    else:
        parts.append(f"RSI neutral ({rsi:.0f})")

    # Price momentum
    ret5 = row.get("return_5d", 0)
    if abs(ret5) > 0.01:
        parts.append(f"Price {ret5:+.1%} over 5d")

    # Volume
    vol_spike = row.get("volume_spike", 1)
    if vol_spike > 1.5:
        parts.append(f"Volume spike {vol_spike:.1f}x avg")

    # MACD
    macd_hist = row.get("macd_hist", 0)
    if abs(macd_hist) > 0.01:
        macd_dir = "bullish crossover" if macd_hist > 0 else "bearish crossover"
        parts.append(f"MACD {macd_dir}")

    signal_label = signal.upper()
    reasoning    = f"{signal_label} — " + ". ".join(parts) if parts else signal_label
    return reasoning


# ── Core prediction ────────────────────────────────────────────────────────────
def predict_signals(
    ticker_filter: str | None = None,
    as_of_date:    str | None = None,
) -> pd.DataFrame:
    """
    Generate signals for the most recent data point per ticker.

    Args:
        ticker_filter : only predict for this ticker (optional)
        as_of_date    : predict as of this date (YYYY-MM-DD). Default: latest available.

    Returns:
        DataFrame with one row per ticker containing signal + metadata
    """
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Features CSV not found: {FEATURES_CSV}. Run feature_engineering.py first."
        )

    model, le, feat_cols = load_artifacts()
    log.info("Model loaded. Classes: %s", list(le.classes_))

    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])

    if ticker_filter:
        df = df[df["ticker"] == ticker_filter.upper()]

    if df.empty:
        log.warning("No data found.")
        return pd.DataFrame()

    # Filter to as-of date
    if as_of_date:
        cutoff = pd.to_datetime(as_of_date)
        df     = df[df["date"] <= cutoff]
    
    # Get the most recent row per ticker
    latest = (
        df.sort_values("date")
        .groupby("ticker")
        .tail(1)
        .reset_index(drop=True)
    )

    log.info("Generating signals for %d tickers as of %s",
             len(latest), latest["date"].max().date() if not latest.empty else "N/A")

    # Fill missing features with 0
    X = latest[feat_cols].fillna(0)

    # Get probabilities for all classes
    probs_array = model.predict_proba(X)
    pred_enc    = model.predict(X)
    pred_labels = le.inverse_transform(pred_enc)

    # Build output
    results = []
    for i, (_, row) in enumerate(latest.iterrows()):
        signal     = pred_labels[i]
        probs      = {cls: round(float(probs_array[i][j]), 4)
                      for j, cls in enumerate(le.classes_)}
        confidence = probs[signal]
        reasoning  = build_reasoning(row, signal)

        results.append({
            "ticker":          row["ticker"],
            "date":            row["date"].strftime("%Y-%m-%d"),
            "signal":          signal,
            "confidence":      round(confidence, 4),
            "prob_buy":        probs.get("buy",  0),
            "prob_hold":       probs.get("hold", 0),
            "prob_sell":       probs.get("sell", 0),
            "sentiment_score": round(row.get("sentiment_score", 0), 4),
            "rsi_14":          round(row.get("rsi_14", 0), 1),
            "return_5d":       round(row.get("return_5d", 0), 4),
            "volume_spike":    round(row.get("volume_spike", 1), 2),
            "reasoning":       reasoning,
        })

    signals_df = pd.DataFrame(results)

    # Sort: buy signals first, then hold, then sell
    order_map  = {"buy": 0, "hold": 1, "sell": 2}
    signals_df["_sort"] = signals_df["signal"].map(order_map)
    signals_df = signals_df.sort_values(["_sort", "confidence"], ascending=[True, False])
    signals_df = signals_df.drop(columns=["_sort"]).reset_index(drop=True)

    return signals_df


def print_signal_table(df: pd.DataFrame) -> None:
    """Pretty-print signals to console."""
    if df.empty:
        log.info("No signals generated.")
        return

    log.info("─" * 80)
    log.info("%-6s  %-5s  %-6s  %-9s  %s", "TICKER", "SIGNAL", "CONF", "SENTIMENT", "REASONING")
    log.info("─" * 80)

    for _, row in df.iterrows():
        signal_icon = {"buy": "▲ BUY ", "sell": "▼ SELL", "hold": "● HOLD"}.get(
            row["signal"], row["signal"]
        )
        log.info(
            "%-6s  %s  %.2f   %+.3f     %s",
            row["ticker"],
            signal_icon,
            row["confidence"],
            row["sentiment_score"],
            row["reasoning"][:60],
        )


def run(
    ticker_filter: str | None = None,
    as_of_date:    str | None = None,
) -> pd.DataFrame:
    signals = predict_signals(ticker_filter, as_of_date)
    print_signal_table(signals)

    signals.to_csv(OUT_CSV, index=False)
    log.info("Signals saved → %s", OUT_CSV)
    return signals


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trading signals")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--date",   type=str, default=None,
                        help="As-of date (YYYY-MM-DD). Default: latest available.")
    args = parser.parse_args()
    run(ticker_filter=args.ticker, as_of_date=args.date)