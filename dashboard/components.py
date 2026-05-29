"""
components.py
-------------
Reusable Streamlit UI components used across all dashboard pages.
Keeping these here avoids repeating layout code across pages.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
MODELS_DIR   = ROOT_DIR / "models"

# ── Colour helpers ─────────────────────────────────────────────────────────────
SIGNAL_BADGE = {
    "buy":  ("▲ BUY",  "#1D9E75", "#0E3D2B"),
    "sell": ("▼ SELL", "#E05C5C", "#3D1515"),
    "hold": ("● HOLD", "#F5A623", "#3D2E0A"),
}


def signal_badge_html(signal: str) -> str:
    label, fg, bg = SIGNAL_BADGE.get(signal.lower(), ("?", "#888", "#222"))
    return (
        f'<span style="'
        f'background:{bg};color:{fg};'
        f'padding:3px 10px;border-radius:6px;'
        f'font-weight:600;font-size:13px;'
        f'border:1px solid {fg}40;'
        f'">{label}</span>'
    )


# ── Metric card row ────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, delta: str | None = None,
                delta_positive: bool = True) -> None:
    """Single metric tile — wraps st.metric with extra styling."""
    delta_colour = "normal" if delta_positive else "inverse"
    st.metric(label=label, value=value, delta=delta, delta_color=delta_colour)


def backtest_metric_row(results: dict) -> None:
    """
    Row of 5 backtest metric cards pulled from backtest_results.json.
    """
    strat = results.get("strategy", {})
    bench = results.get("benchmark", {})
    alpha = results.get("alpha", 0)

    total_ret  = strat.get("total_return",      0)
    ann_ret    = strat.get("annualised_return",  0)
    sharpe     = strat.get("sharpe_ratio",       0)
    mdd        = strat.get("max_drawdown",       0)
    win_rate   = strat.get("win_rate",           0)
    b_ret      = bench.get("total_return",       0)

    cols = st.columns(5)
    with cols[0]:
        metric_card(
            "Total Return",
            f"{total_ret:+.1%}",
            delta=f"{alpha:+.1%} vs B&H",
            delta_positive=alpha >= 0,
        )
    with cols[1]:
        metric_card(
            "Ann. Return",
            f"{ann_ret:+.1%}",
        )
    with cols[2]:
        metric_card(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            delta="↑ good > 1.0",
            delta_positive=sharpe >= 1.0,
        )
    with cols[3]:
        metric_card(
            "Max Drawdown",
            f"{mdd:.1%}",
            delta_positive=False,
        )
    with cols[4]:
        metric_card(
            "Win Rate",
            f"{win_rate:.1%}",
            delta_positive=win_rate >= 0.5,
        )


# ── Signal summary cards ───────────────────────────────────────────────────────
def signal_summary_cards(signals_df: pd.DataFrame) -> None:
    """
    Three stat cards: # BUY, # HOLD, # SELL signals today.
    """
    counts = signals_df["signal"].value_counts()
    n_buy  = int(counts.get("buy",  0))
    n_hold = int(counts.get("hold", 0))
    n_sell = int(counts.get("sell", 0))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""<div style="background:#0E3D2B;border:1px solid #1D9E7540;
            border-radius:10px;padding:14px 18px;text-align:center">
            <div style="font-size:28px;font-weight:700;color:#1D9E75">{n_buy}</div>
            <div style="font-size:13px;color:#aaa;margin-top:4px">▲ BUY Signals</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div style="background:#3D2E0A;border:1px solid #F5A62340;
            border-radius:10px;padding:14px 18px;text-align:center">
            <div style="font-size:28px;font-weight:700;color:#F5A623">{n_hold}</div>
            <div style="font-size:13px;color:#aaa;margin-top:4px">● HOLD Signals</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div style="background:#3D1515;border:1px solid #E05C5C40;
            border-radius:10px;padding:14px 18px;text-align:center">
            <div style="font-size:28px;font-weight:700;color:#E05C5C">{n_sell}</div>
            <div style="font-size:13px;color:#aaa;margin-top:4px">▼ SELL Signals</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ── Reasoning card ─────────────────────────────────────────────────────────────
def reasoning_card(row: pd.Series) -> None:
    """
    Renders the AI reasoning text for a single signal in a styled card.
    """
    signal = row.get("signal", "hold")
    _, fg, bg = SIGNAL_BADGE.get(signal.lower(), ("?", "#888", "#222"))
    badge_html = signal_badge_html(signal)
    reasoning  = row.get("reasoning", "No reasoning available.")
    ticker     = row.get("ticker", "")
    conf       = float(row.get("confidence", 0))
    sent       = float(row.get("sentiment_score", 0))

    st.markdown(
        f"""<div style="background:#1A1D27;border:1px solid {fg}30;
        border-radius:10px;padding:16px 20px;margin-bottom:10px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <span style="font-size:16px;font-weight:600;color:#E0E0E0">{ticker}</span>
            {badge_html}
        </div>
        <div style="font-size:13px;color:#B0B0B0;margin-bottom:8px">{reasoning}</div>
        <div style="display:flex;gap:20px;font-size:12px;color:#888">
            <span>Confidence: <b style="color:#E0E0E0">{conf:.1%}</b></span>
            <span>Sentiment: <b style="color:{fg}">{sent:+.3f}</b></span>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)   # cache for 5 minutes
def load_signals() -> pd.DataFrame:
    path = DATA_DIR / "latest_signals.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_daily_sentiment() -> pd.DataFrame:
    path = DATA_DIR / "daily_sentiment.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        return df
    return pd.DataFrame()


@st.cache_data(ttl=600)
def load_prices() -> pd.DataFrame:
    import sqlite3
    db = DATA_DIR / "raw_prices.db"
    if not db.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db)
    df   = pd.read_sql("SELECT * FROM prices", conn, parse_dates=["date"])
    conn.close()
    return df


@st.cache_data(ttl=600)
def load_backtest_results() -> dict:
    path = DATA_DIR / "backtest_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=600)
def load_feature_importance() -> pd.DataFrame:
    """Load SHAP global importance if available."""
    import pickle, json
    feat_path = MODELS_DIR / "feature_columns.json"
    model_path = MODELS_DIR / "xgb_signal_model.pkl"

    if not model_path.exists() or not feat_path.exists():
        return pd.DataFrame()

    try:
        import shap
        import numpy as np
        from ml.train import load_artifacts

        model, le, feat_cols = load_artifacts()
        features_csv = DATA_DIR / "features.csv"
        if not features_csv.exists():
            return pd.DataFrame()

        df = pd.read_csv(features_csv).sample(min(300, len(pd.read_csv(features_csv))),
                                               random_state=42)
        X  = df[feat_cols].fillna(0)

        from sklearn.ensemble import VotingClassifier
        from xgboost import XGBClassifier
        if isinstance(model, VotingClassifier):
            for name, est in model.estimators_:
                if isinstance(est, XGBClassifier):
                    model = est
                    break
        explainer = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)

        if isinstance(shap_vals, list):
            mean_abs = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            mean_abs = np.abs(shap_vals)

        importance = pd.DataFrame({
            "feature":    feat_cols,
            "importance": mean_abs.mean(axis=0),
        }).sort_values("importance", ascending=False)

        return importance
    except Exception:
        return pd.DataFrame()


# ── Page header ────────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""<div style="margin-bottom:20px">
        <h2 style="margin:0;font-size:22px;font-weight:700;color:#E0E0E0">{title}</h2>
        {"<p style='margin:4px 0 0;font-size:13px;color:#888'>" + subtitle + "</p>" if subtitle else ""}
        </div>""",
        unsafe_allow_html=True,
    )


# ── Empty state ────────────────────────────────────────────────────────────────
def empty_state(message: str, command: str) -> None:
    st.info(
        f"**{message}**\n\nRun `{command}` to generate data, then refresh.",
        icon="ℹ️",
    )


# ── Sidebar ticker selector ────────────────────────────────────────────────────
def ticker_selector(signals_df: pd.DataFrame, label: str = "Select ticker") -> str | None:
    tickers = sorted(signals_df["ticker"].unique().tolist()) if not signals_df.empty else []
    if not tickers:
        return None
    return st.sidebar.selectbox(label, tickers)