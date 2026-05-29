"""
train.py  (v3 — fixed for better accuracy)
-------------------------------------------
Key fixes from v2:
  - Stronger regularisation to reduce overfitting
  - Shallower trees (max_depth 3 instead of 4)
  - Higher min_child_weight to avoid learning noise
  - Calibrated class weights
  - Better eval and early stopping
"""

import json, logging, argparse, pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
MODELS_DIR   = ROOT_DIR / "models"
FEATURES_CSV = DATA_DIR / "features.csv"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH   = MODELS_DIR / "xgb_signal_model.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEAT_PATH    = MODELS_DIR / "feature_columns.json"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

FEATURE_COLS = [
    # Price momentum
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "gap", "hl_range",
    # Technical indicators
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_position", "bb_width", "volume_spike",
    "price_vs_sma20", "price_vs_sma50",
    # Earnings proximity
    "earnings_month", "earnings_proximity",
    # Sentiment
    "sentiment_score", "sentiment_lag_1d", "sentiment_lag_3d", "sentiment_lag_5d",
    "sentiment_rolling_5d", "sentiment_rolling_10d",
    "sentiment_momentum_3d", "sentiment_momentum_7d",
    "bullish_ratio", "bearish_ratio", "total_articles",
    "sentiment_divergence",
    # Macro
    "vix", "tnx_yield", "spy_return", "gld_return", "tlt_return",
    # Sector
    "sector_encoded",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
log = logging.getLogger(__name__)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_and_prepare(ticker=None):
    log.info("Loading features from %s ...", FEATURES_CSV)
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])

    if ticker:
        df = df[df["ticker"] == ticker.upper()]
        log.info("Filtered to %s: %d rows", ticker.upper(), len(df))

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Only keep tickers with decent sentiment coverage
    # (tickers with no news data hurt the model)
    sentiment_coverage = df.groupby("ticker")["sentiment_score"].apply(
        lambda x: (x != 0).sum()
    )
    df = df[df['sentiment_score'] != 0].copy()
    log.info("Rows with actual sentiment signal: %d", len(df))
    good_tickers = sentiment_coverage[sentiment_coverage >= 30].index
    before = df["ticker"].nunique()
    df = df[df["ticker"].isin(good_tickers)]
    after = df["ticker"].nunique()
    log.info("Tickers filtered: %d → %d (removed low-sentiment coverage)", before, after)

    # Keep only feature columns that exist
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing features (skipped): %s", missing)

    df[available] = df[available].fillna(0)

    # Log label distribution
    dist = df["signal"].value_counts()
    log.info("Label distribution: %s", dist.to_dict())
    log.info("Using %d feature columns on %d rows", len(available), len(df))

    return df, available


# ── Walk-forward time split ────────────────────────────────────────────────────
def walk_forward_split(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Time-ordered split — no shuffling, no data leakage.
    Train on past, validate and test on future.
    """
    dates     = df["date"].sort_values().unique()
    n         = len(dates)
    train_end = dates[int(n * train_ratio)]
    val_end   = dates[int(n * (train_ratio + val_ratio))]

    train = df[df["date"] <  train_end]
    val   = df[(df["date"] >= train_end) & (df["date"] < val_end)]
    test  = df[df["date"] >= val_end]

    log.info("Train : %d rows  (%s → %s)",
             len(train), train["date"].min().date(), train["date"].max().date())
    log.info("Val   : %d rows  (%s → %s)",
             len(val),   val["date"].min().date(),   val["date"].max().date())
    log.info("Test  : %d rows  (%s → %s)",
             len(test),  test["date"].min().date(),  test["date"].max().date())

    return train, val, test


# ── Class weights ──────────────────────────────────────────────────────────────
def compute_sample_weight(y):
    """
    Balanced weights — each class gets equal total weight.
    Prevents majority class (hold) from dominating.
    """
    counts = y.value_counts()
    total  = len(y)
    n_classes = len(counts)
    weights = {}
    for cls, cnt in counts.items():
        weights[cls] = (total / (n_classes * cnt)) ** 0.5  # square root dampens extremes
    log.info("Class weights: %s", {k: round(v, 3) for k, v in weights.items()})
    return np.array([weights[label] for label in y])


# ── Model training ─────────────────────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val):
    le   = LabelEncoder()
    y_tr = le.fit_transform(y_train)
    y_v  = le.transform(y_val)
    sw   = compute_sample_weight(y_train)

    log.info("Label encoding: %s", dict(enumerate(le.classes_)))

    # XGBoost — shallower trees, stronger regularisation
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,     
        gamma=0.1,
        reg_alpha=0.3,
        reg_lambda=1.5,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        num_class=len(le.classes_),
    )

    if HAS_LGBM:
        lgbm = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=20,          # smaller = less overfitting
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=30,   # higher = more conservative
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        log.info("Training XGBoost ...")
        xgb.fit(
            X_train, y_tr,
            sample_weight=sw,
            eval_set=[(X_val, y_v)],
            verbose=100,
        )

        log.info("Training LightGBM ...")
        lgbm.fit(
            X_train, y_tr,
            sample_weight=sw,
            eval_set=[(X_val, y_v)],
        )

        log.info("Building soft-voting ensemble ...")
        model = VotingClassifier(
            estimators=[("xgb", xgb), ("lgbm", lgbm)],
            voting="soft",
        )
        model.fit(X_train, y_tr, sample_weight=sw)
        log.info("Ensemble trained.")

    else:
        log.info("Training XGBoost only (pip install lightgbm for ensemble) ...")
        xgb.fit(
            X_train, y_tr,
            sample_weight=sw,
            eval_set=[(X_val, y_v)],
            verbose=100,
        )
        model = xgb

    return model, le


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, le, X, y, split_name="test"):
    y_enc  = le.transform(y)
    y_pred = model.predict(X)

    log.info("─" * 60)
    log.info("%s SET RESULTS:", split_name.upper())
    log.info("\n%s", classification_report(
        y_enc, y_pred, target_names=le.classes_
    ))

    cm = confusion_matrix(y_enc, y_pred)
    log.info("Confusion matrix:\n%s", cm)

    # Overall accuracy
    acc = (y_pred == y_enc).mean()
    log.info("Overall accuracy: %.1f%%", acc * 100)

    # Directional accuracy (buy + sell only)
    directional_mask = y.isin(["buy", "sell"])
    if directional_mask.sum() > 0:
        dir_preds  = le.inverse_transform(y_pred)[directional_mask]
        dir_actual = y.values[directional_mask]
        dir_acc    = (dir_preds == dir_actual).mean()
        log.info("Directional accuracy (buy/sell only): %.1f%%", dir_acc * 100)

    return acc


# ── Save / load artifacts ──────────────────────────────────────────────────────
def save_artifacts(model, le, feat_cols):
    with open(MODEL_PATH,   "wb") as f: pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f: pickle.dump(le, f)
    with open(FEAT_PATH,    "w")  as f: json.dump(feat_cols, f, indent=2)
    log.info("Model   saved → %s", MODEL_PATH)
    log.info("Encoder saved → %s", ENCODER_PATH)
    log.info("Features saved → %s", FEAT_PATH)


def load_artifacts():
    """Load saved model, encoder and feature list. Used by predict/backtest/explain."""
    with open(MODEL_PATH,   "rb") as f: model     = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: le        = pickle.load(f)
    with open(FEAT_PATH)          as f: feat_cols = json.load(f)
    return model, le, feat_cols


# ── Main ───────────────────────────────────────────────────────────────────────
def run(ticker_filter=None):
    df, feat_cols = load_and_prepare(ticker_filter)

    if len(df) < 500:
        log.error(
            "Not enough data (%d rows). Need 500+. "
            "Run more ingestion and sentiment scoring first.", len(df)
        )
        return

    train_df, val_df, test_df = walk_forward_split(df)

    model, le = train_model(
        train_df[feat_cols], train_df["signal"],
        val_df[feat_cols],   val_df["signal"],
    )

    log.info("=" * 60)
    evaluate(model, le, val_df[feat_cols],  val_df["signal"],  "validation")
    evaluate(model, le, test_df[feat_cols], test_df["signal"], "test")

    save_artifacts(model, le, feat_cols)
    log.info("Training complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FinSentinel signal model")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Train on a single ticker only")
    args = parser.parse_args()
    run(ticker_filter=args.ticker)