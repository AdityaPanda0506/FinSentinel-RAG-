"""
app.py
------
FinSentinel — AI Market Sentiment Signal Engine
Main Streamlit dashboard entry point.

Pages:
    1. Overview       — signal summary, heatmap, key metrics
    2. Live Signals   — full signal table with reasoning
    3. Ticker Deep Dive — candlestick + sentiment for one ticker
    4. Backtest       — equity curve + full performance metrics
    5. Explainability — SHAP feature importance charts

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Make project root importable from any working directory
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="FinSentinel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide Streamlit default header & footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* Tighten default padding */
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    /* Metric delta text size */
    [data-testid="stMetricDelta"] { font-size: 11px; }

    /* Subtle dividers */
    hr { border-color: rgba(255,255,255,0.07); margin: 0.8rem 0; }

    /* Sidebar nav */
    .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

from dashboard.components import (
    load_signals,
    load_daily_sentiment,
    load_prices,
    load_backtest_results,
    load_feature_importance,
    page_header,
    empty_state,
    backtest_metric_row,
    signal_summary_cards,
    reasoning_card,
    ticker_selector,
)
from dashboard.charts import (
    sentiment_heatmap,
    signal_table,
    equity_curve,
    price_sentiment_chart,
    sentiment_distribution,
    feature_importance_chart,
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """<div style="padding:10px 0 20px">
        <span style="font-size:22px;font-weight:700;color:#185FA5">📈 FinSentinel</span>
        <div style="font-size:11px;color:#666;margin-top:3px">
        AI Market Sentiment Engine
        </div></div>""",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Overview", "Live Signals", "Ticker Deep Dive",
         "Backtest Results", "Explainability"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick pipeline runner buttons
    st.markdown(
        "<div style='font-size:11px;color:#666;margin-bottom:6px'>"
        "PIPELINE CONTROLS</div>",
        unsafe_allow_html=True,
    )

    if st.button("🔄  Refresh Signals", use_container_width=True):
        import subprocess, os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        with st.spinner("Generating signals ..."):
            result = subprocess.run(
                ["python", str(ROOT / "ml" / "predict.py")],
                capture_output=True, text=True, cwd=str(ROOT), env=env,
            )
        if result.returncode == 0:
            st.success("Signals refreshed!")
            st.cache_data.clear()
        else:
            st.error(f"Error: {result.stderr[:200]}")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:10px;color:#444;text-align:center'>"
        "Data: NewsAPI · yFinance · Reddit<br>"
        "Model: XGBoost + FinBERT<br>"
        "SHAP Explainability"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Load shared data ───────────────────────────────────────────────────────────
signals_df   = load_signals()
sentiment_df = load_daily_sentiment()
prices_df    = load_prices()
backtest_res = load_backtest_results()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    page_header(
        "Market Overview",
        f"Sentiment-driven signals across {len(signals_df['ticker'].unique()) if not signals_df.empty else 0} tickers",
    )

    # ── Signal summary cards ───────────────────────────────────────────────────
    if not signals_df.empty:
        signal_summary_cards(signals_df)
    else:
        empty_state("No signals yet", "python ml/predict.py")

    st.markdown("---")

    # ── Backtest metrics row ───────────────────────────────────────────────────
    if backtest_res:
        st.markdown("#### Backtest Performance")
        backtest_metric_row(backtest_res)
    else:
        st.info("Run `python ml/backtest.py` to see performance metrics.", icon="ℹ️")

    st.markdown("---")

    # ── Sentiment heatmap ──────────────────────────────────────────────────────
    if not sentiment_df.empty:
        st.markdown("#### Sentiment Heatmap")
        fig = sentiment_heatmap(sentiment_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        empty_state("No sentiment data", "python nlp/aggregator.py")

    # ── Sentiment distribution ─────────────────────────────────────────────────
    if not sentiment_df.empty:
        st.markdown("#### Article Sentiment by Ticker")
        fig2 = sentiment_distribution(sentiment_df)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Signals":
    page_header(
        "Live Signal Feed",
        "Buy / Hold / Sell signals with AI reasoning for each ticker",
    )

    if signals_df.empty:
        empty_state("No signals found", "python ml/predict.py")
        st.stop()

    # ── Filter controls ────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        signal_filter = st.selectbox("Signal type", ["All", "BUY", "HOLD", "SELL"])
    with col_f2:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.5, 0.05)
    with col_f3:
        search = st.text_input("Search ticker", placeholder="e.g. AAPL, TSLA")

    filtered = signals_df.copy()
    if signal_filter != "All":
        filtered = filtered[filtered["signal"] == signal_filter.lower()]
    filtered = filtered[filtered["confidence"] >= min_conf]
    if search.strip():
        filtered = filtered[
            filtered["ticker"].str.upper().str.contains(search.upper())
        ]

    st.markdown(f"**{len(filtered)} signals** matching filters")
    st.markdown("---")

    # ── Signal table ───────────────────────────────────────────────────────────
    fig = signal_table(filtered)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Reasoning cards ────────────────────────────────────────────────────────
    st.markdown("#### Signal Reasoning")
    st.caption("AI-generated explanation for each signal based on sentiment + technicals")

    # Show top 10 highest-confidence signals
    top_signals = filtered.sort_values("confidence", ascending=False).head(10)
    for _, row in top_signals.iterrows():
        reasoning_card(row)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TICKER DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Ticker Deep Dive":
    page_header("Ticker Deep Dive", "Price chart, sentiment trend, and signal history")

    # ── Ticker selector ────────────────────────────────────────────────────────
    col_t, col_d = st.columns([1, 1])
    with col_t:
        if not signals_df.empty:
            available_tickers = sorted(signals_df["ticker"].unique().tolist())
        elif not sentiment_df.empty:
            available_tickers = sorted(sentiment_df["ticker"].unique().tolist())
        else:
            available_tickers = []

        if not available_tickers:
            empty_state("No data available", "python ingestion/price_fetcher.py")
            st.stop()

        selected_ticker = st.selectbox("Select ticker", available_tickers)

    with col_d:
        chart_days = st.selectbox("Chart period", [30, 60, 90, 180], index=2)

    # ── Current signal card ────────────────────────────────────────────────────
    if not signals_df.empty:
        ticker_signal = signals_df[signals_df["ticker"] == selected_ticker]
        if not ticker_signal.empty:
            st.markdown("#### Current Signal")
            reasoning_card(ticker_signal.iloc[0])

    # ── Price + sentiment chart ────────────────────────────────────────────────
    if not prices_df.empty and not sentiment_df.empty:
        st.markdown("#### Price & Sentiment")
        fig = price_sentiment_chart(
            prices_df, sentiment_df, selected_ticker, days=chart_days
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        empty_state("No price data", "python ingestion/price_fetcher.py")

    # ── Sentiment time series ──────────────────────────────────────────────────
    if not sentiment_df.empty:
        ticker_sent = sentiment_df[sentiment_df["ticker"] == selected_ticker].copy()
        ticker_sent = ticker_sent.sort_values("date").tail(chart_days)

        if not ticker_sent.empty:
            st.markdown("#### Sentiment Score Over Time")

            import plotly.graph_objects as go
            fig2 = go.Figure()

            # Zero line
            fig2.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)

            # Sentiment bars coloured by direction
            colours = [
                "#1D9E75" if s > 0.05 else
                "#E05C5C" if s < -0.05 else
                "#888888"
                for s in ticker_sent["sentiment_score"]
            ]
            fig2.add_trace(go.Bar(
                x=ticker_sent["date"],
                y=ticker_sent["sentiment_score"],
                marker_color=colours,
                marker_opacity=0.8,
                name="Daily Sentiment",
                hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
            ))

            # 7-day rolling average
            rolling = ticker_sent["sentiment_score"].rolling(7, min_periods=1).mean()
            fig2.add_trace(go.Scatter(
                x=ticker_sent["date"],
                y=rolling,
                mode="lines",
                name="7d avg",
                line=dict(color="#F5A623", width=2),
            ))

            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E0E0E0"),
                margin=dict(l=10, r=10, t=30, b=10),
                height=280,
                legend=dict(x=0.01, y=0.99),
                yaxis=dict(
                    title="Sentiment Score",
                    range=[-1.1, 1.1],
                    gridcolor="rgba(255,255,255,0.06)",
                ),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Key stats table ────────────────────────────────────────────────────────
    if not signals_df.empty:
        ticker_row = signals_df[signals_df["ticker"] == selected_ticker]
        if not ticker_row.empty:
            r = ticker_row.iloc[0]
            st.markdown("#### Key Statistics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sentiment Score",  f"{r.get('sentiment_score', 0):+.3f}")
            c2.metric("RSI (14d)",         f"{r.get('rsi_14', 0):.1f}")
            c3.metric("5d Return",         f"{r.get('return_5d', 0):+.1%}")
            c4.metric("Volume Spike",      f"{r.get('volume_spike', 1):.1f}x")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BACKTEST RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Backtest Results":
    page_header(
        "Backtest Results",
        "Out-of-sample strategy performance vs buy-and-hold benchmark",
    )

    if not backtest_res:
        empty_state("No backtest results", "python ml/backtest.py --plot")
        st.stop()

    # ── Metrics ────────────────────────────────────────────────────────────────
    st.markdown("#### Strategy vs Benchmark")
    backtest_metric_row(backtest_res)

    st.markdown("---")

    # ── Strategy vs Benchmark comparison table ─────────────────────────────────
    strat = backtest_res.get("strategy", {})
    bench = backtest_res.get("benchmark", {})

    compare = {
        "Metric":             ["Total Return", "Annualised Return", "Sharpe Ratio",
                               "Sortino Ratio", "Max Drawdown", "Win Rate", "Calmar Ratio"],
        "FinSentinel":        [
            f"{strat.get('total_return',0):+.1%}",
            f"{strat.get('annualised_return',0):+.1%}",
            f"{strat.get('sharpe_ratio',0):.3f}",
            f"{strat.get('sortino_ratio',0):.3f}",
            f"{strat.get('max_drawdown',0):.1%}",
            f"{strat.get('win_rate',0):.1%}",
            f"{strat.get('calmar_ratio',0):.3f}",
        ],
        "Buy & Hold":         [
            f"{bench.get('total_return',0):+.1%}",
            f"{bench.get('annualised_return',0):+.1%}",
            f"{bench.get('sharpe_ratio',0):.3f}",
            f"{bench.get('sortino_ratio',0):.3f}",
            f"{bench.get('max_drawdown',0):.1%}",
            "—",
            f"{bench.get('calmar_ratio',0):.3f}",
        ],
    }

    import pandas as pd
    compare_df = pd.DataFrame(compare).set_index("Metric")
    st.dataframe(compare_df, use_container_width=True)

    st.markdown("---")

    # ── Equity curve ───────────────────────────────────────────────────────────
    st.markdown("#### Equity Curve")
    fig = equity_curve(backtest_res)
    st.plotly_chart(fig, use_container_width=True)

    # ── Static equity curve image (from backtest.py --plot) ───────────────────
    equity_img = ROOT / "models" / "equity_curve.png"
    if equity_img.exists():
        st.markdown("#### Detailed Equity Curve (from backtest.py)")
        st.image(str(equity_img), use_container_width=True)

    st.markdown("---")

    # ── Trade log sample ───────────────────────────────────────────────────────
    trade_log = backtest_res.get("trade_log", [])
    if trade_log:
        st.markdown("#### Sample Trade Log")
        trade_df = pd.DataFrame(trade_log)
        trade_df["return"] = (trade_df["return"] * 100).round(2).astype(str) + "%"
        trade_df["profitable"] = trade_df["profitable"].map({True: "✓", False: "✗"})
        st.dataframe(
            trade_df.rename(columns={
                "ticker": "Ticker", "entry_date": "Entry", "exit_date": "Exit",
                "entry_price": "Entry $", "exit_price": "Exit $",
                "return": "Return", "profitable": "Win",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Explainability":
    page_header(
        "Model Explainability",
        "SHAP analysis — what drives each signal decision",
    )

    # ── Global importance ──────────────────────────────────────────────────────
    st.markdown("#### Global Feature Importance")
    st.caption(
        "Mean absolute SHAP value per feature — higher = more influential in the model's decisions"
    )

    # Try loading pre-saved SHAP chart first (fast)
    shap_img = ROOT / "models" / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), use_container_width=True)
    else:
        # Compute live (slower)
        with st.spinner("Computing SHAP values ..."):
            importance_df = load_feature_importance()

        if not importance_df.empty:
            fig = feature_importance_chart(importance_df, top_n=15)
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state(
                "No model found",
                "python ml/train.py && python ml/explain.py"
            )

    st.markdown("---")

    # ── Per-ticker SHAP waterfall ──────────────────────────────────────────────
    st.markdown("#### Per-Ticker Signal Explanation")
    st.caption("Why did the model generate this signal for this ticker?")

    col_tick, col_btn = st.columns([2, 1])
    with col_tick:
        explain_ticker = st.selectbox(
            "Choose ticker to explain",
            sorted(signals_df["ticker"].unique().tolist()) if not signals_df.empty else [],
        )
    with col_btn:
        st.markdown("<div style='margin-top:28px'>", unsafe_allow_html=True)
        run_explain = st.button("Generate Explanation", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Check for pre-generated waterfall image
    if explain_ticker:
        ticker_img = ROOT / "models" / f"shap_{explain_ticker.lower()}.png"
        if ticker_img.exists():
            st.image(str(ticker_img), use_container_width=True)
        elif run_explain:
            import subprocess
            with st.spinner(f"Computing SHAP for {explain_ticker} ..."):
                import os
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT)
                result = subprocess.run(
                    ["python", str(ROOT / "ml" / "explain.py"),
                     "--ticker", explain_ticker],
                    capture_output=True, text=True, cwd=str(ROOT), env=env,
                )
            if result.returncode == 0 and ticker_img.exists():
                st.image(str(ticker_img), use_container_width=True)
            else:
                st.error(
                    f"Could not generate explanation.\n"
                    f"Run manually: `python ml/explain.py --ticker {explain_ticker}`"
                )
        else:
            st.info(
                f"Click **Generate Explanation** to compute SHAP for {explain_ticker}.",
                icon="ℹ️",
            )

    st.markdown("---")

    # ── How to read SHAP explanation ──────────────────────────────────────────
    with st.expander("How to read SHAP charts"):
        st.markdown("""
**SHAP (SHapley Additive exPlanations)** measures how much each feature
contributes to pushing the model's output higher or lower.

- **Global chart** — bars show average impact across all predictions.
  Long bar = feature matters a lot. Short bar = feature barely influences the model.

- **Waterfall chart** — for one specific ticker prediction.
  - 🟢 **Green bars** push toward BUY (positive SHAP value)
  - 🔴 **Red bars** push toward SELL (negative SHAP value)
  - The sum of all bars = the model's final output score

**Example interpretation:**
> *RSI at 32 pushed the model +0.18 toward BUY (oversold signal).
> Sentiment score of +0.45 added another +0.14.
> Volume spike of 2.1x added +0.09.
> Result: BUY with 78% confidence.*
        """)