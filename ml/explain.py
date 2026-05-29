"""
explain.py
----------
Uses SHAP (SHapley Additive exPlanations) to explain:
    1. Global feature importance — which features drive the model overall
    2. Per-ticker explanation    — why did the model signal BUY/SELL for AAPL today?

SHAP is the gold standard for ML explainability and is heavily
used in quant finance. Adding this to your project immediately
impresses interviewers who know the field.

Usage:
    python ml/explain.py                        # global importance plot
    python ml/explain.py --ticker AAPL          # explain latest AAPL prediction
    python ml/explain.py --ticker TSLA --top 10 # top 10 features for TSLA

Output:
    models/shap_summary.png       — global feature importance bar chart
    models/shap_<ticker>.png      — waterfall plot for a specific ticker
"""

import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import shap

from ml.train import load_artifacts

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
MODELS_DIR   = ROOT_DIR / "models"
FEATURES_CSV = DATA_DIR / "features.csv"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data loader ────────────────────────────────────────────────────────────────
def load_features(ticker: str | None = None, sample_size: int = 500) -> pd.DataFrame:
    """
    Load feature matrix. Optionally sample to keep SHAP computation fast.
    SHAP on 500 rows is representative and runs in seconds.
    """
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    if ticker:
        df = df[df["ticker"] == ticker.upper()]

    df = df.sort_values("date").reset_index(drop=True)

    # For global explanation, sample evenly across time
    if len(df) > sample_size and not ticker:
        df = df.sample(n=sample_size, random_state=42).sort_values("date")
        log.info("Sampled %d rows for SHAP computation", sample_size)

    return df


# ── SHAP explainer ─────────────────────────────────────────────────────────────
def build_explainer(model, X: pd.DataFrame) -> shap.Explainer:
    """
    Build a SHAP TreeExplainer.
    Handles XGBoost directly or extracts it from VotingClassifier.
    """
    from sklearn.ensemble import VotingClassifier
    from xgboost import XGBClassifier
    log.info("Building SHAP TreeExplainer ...")

    # Already a plain XGBoost model
    if isinstance(model, XGBClassifier):
        log.info("Model is XGBoost — building TreeExplainer directly")
        return shap.TreeExplainer(model)

    # VotingClassifier ensemble — extract XGBoost from it
    if isinstance(model, VotingClassifier):
        log.info("Extracting XGBoost from VotingClassifier ensemble ...")
        for item in model.estimators_:
            # estimators_ can be list of estimators or list of (name, est) tuples
            est = item[1] if isinstance(item, tuple) else item
            if isinstance(est, XGBClassifier):
                log.info("Found XGBoost in ensemble")
                return shap.TreeExplainer(est)
        # Fallback to first estimator
        first = model.estimators_[0]
        est   = first[1] if isinstance(first, tuple) else first
        return shap.TreeExplainer(est)

    # Generic fallback
    log.info("Using generic TreeExplainer")
    return shap.TreeExplainer(model)


def compute_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values. Returns array of shape (n_samples, n_features, n_classes).
    """
    log.info("Computing SHAP values for %d rows ...", len(X))
    shap_values = explainer.shap_values(X)
    return shap_values


# ── Global feature importance ──────────────────────────────────────────────────
def plot_global_importance(
    shap_values: np.ndarray,
    X:           pd.DataFrame,
    le,
    top_n:       int = 15,
    save_path:   Path | None = None,
) -> pd.DataFrame:
    """
    Bar chart of mean |SHAP| per feature, averaged across all classes.
    Most important features across the entire model.
    """
    # Average absolute SHAP across all classes
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        # shape: (n_samples, n_features, n_classes) — average across classes
        mean_abs = np.abs(shap_values).mean(axis=2)
    else:
        mean_abs = np.abs(shap_values)

    importance = pd.DataFrame({
        "feature":    X.columns,
        "importance": mean_abs.mean(axis=0),
    }).sort_values("importance", ascending=False)

    log.info("Top %d features by mean |SHAP|:", top_n)
    for _, row in importance.head(top_n).iterrows():
        bar = "█" * int(row["importance"] * 200)
        log.info("  %-30s  %.4f  %s", row["feature"], row["importance"], bar[:30])

    # Plot
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(top["feature"][::-1], top["importance"][::-1],
                   color="#185FA5", alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("FinSentinel — Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, top["importance"][::-1]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()

    if save_path is None:
        save_path = MODELS_DIR / "shap_summary.png"

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Global SHAP plot saved → %s", save_path)

    return importance


# ── Per-ticker waterfall explanation ──────────────────────────────────────────
def explain_ticker(
    ticker:      str,
    explainer,
    model,
    le,
    feat_cols:   list[str],
    top_n:       int = 10,
) -> dict:
    """
    Explain the latest prediction for a specific ticker.
    Shows which features pushed the model towards buy/hold/sell.

    Returns a dict with the explanation data (also used by dashboard).
    """
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    df = df[df["ticker"] == ticker.upper()].sort_values("date")

    if df.empty:
        log.warning("No data found for ticker %s", ticker)
        return {}

    latest    = df.tail(1)
    X_latest  = latest[feat_cols].fillna(0)
    date_str  = latest["date"].iloc[0].strftime("%Y-%m-%d")

    # Get prediction
    pred_enc   = model.predict(X_latest)[0]
    pred_label = le.inverse_transform([pred_enc])[0]
    probs      = model.predict_proba(X_latest)[0]
    prob_dict  = {cls: float(probs[j]) for j, cls in enumerate(le.classes_)}

    log.info("Explaining %s prediction on %s: %s (conf: %.2f)",
             ticker, date_str, pred_label.upper(), prob_dict[pred_label])

    # SHAP values for this single row
    sv = explainer.shap_values(X_latest)

    # Handle all possible SHAP output shapes
    class_idx = list(le.classes_).index(pred_label)
    if isinstance(sv, list):
        # list of (n_samples, n_features) — one per class
        sv_for_class = sv[class_idx][0]
    elif sv.ndim == 3:
        # shape: (n_samples, n_features, n_classes)
        sv_for_class = sv[0, :, class_idx]
    elif sv.ndim == 2:
        # shape: (n_samples, n_features) — binary or single output
        sv_for_class = sv[0]
    else:
        sv_for_class = sv[0]

    # Build explanation DataFrame
    explanation = pd.DataFrame({
        "feature":     feat_cols,
        "shap_value":  sv_for_class,
        "feature_val": X_latest.values[0],
    }).sort_values("shap_value", key=abs, ascending=False)

    log.info("Top %d drivers for %s → %s:", top_n, ticker, pred_label.upper())
    for _, row in explanation.head(top_n).iterrows():
        direction = "▲" if row["shap_value"] > 0 else "▼"
        log.info("  %s %-30s  shap=%+.4f  value=%.4f",
                 direction, row["feature"], row["shap_value"], row["feature_val"])

    # Waterfall plot
    fig, ax = plt.subplots(figsize=(10, 7))

    top_exp  = explanation.head(top_n).iloc[::-1]
    colors   = ["#1D9E75" if v > 0 else "#E05C5C" for v in top_exp["shap_value"]]
    bars = ax.barh(top_exp["feature"], top_exp["shap_value"], color=colors, alpha=0.85)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on model output)", fontsize=11)
    ax.set_title(
        f"FinSentinel — Why {ticker} → {pred_label.upper()}  ({date_str})\n"
        f"Confidence: {prob_dict[pred_label]:.1%}",
        fontsize=13, fontweight="bold"
    )
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, top_exp["shap_value"]):
        ax.text(
            bar.get_width() + (0.0002 if val >= 0 else -0.0002),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )

    plt.tight_layout()
    save_path = MODELS_DIR / f"shap_{ticker.lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Ticker SHAP plot saved → %s", save_path)

    return {
        "ticker":      ticker,
        "date":        date_str,
        "signal":      pred_label,
        "confidence":  prob_dict[pred_label],
        "probabilities": prob_dict,
        "top_features":  explanation.head(top_n).to_dict(orient="records"),
        "plot_path":     str(save_path),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def run(ticker_filter: str | None = None, top_n: int = 15) -> None:
    model, le, feat_cols = load_artifacts()

    df = load_features(ticker=ticker_filter)
    X  = df[feat_cols].fillna(0)

    explainer  = build_explainer(model, X)
    shap_vals  = compute_shap_values(explainer, X)

    # Always produce global plot
    plot_global_importance(shap_vals, X, le, top_n=top_n)

    # If specific ticker requested, produce waterfall plot too
    if ticker_filter:
        explain_ticker(ticker_filter, explainer, model, le, feat_cols, top_n=top_n)

    log.info("SHAP analysis complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP explainability for signal model")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Generate waterfall explanation for a specific ticker")
    parser.add_argument("--top",    type=int, default=15,
                        help="Number of top features to show (default: 15)")
    args = parser.parse_args()
    run(ticker_filter=args.ticker, top_n=args.top)