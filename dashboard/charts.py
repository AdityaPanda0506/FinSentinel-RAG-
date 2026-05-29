"""
charts.py
---------
All Plotly chart builder functions used by the Streamlit dashboard.
Each function takes a DataFrame and returns a go.Figure object.

Keeping charts in a separate file keeps app.py clean and
makes individual charts easy to test and reuse.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE      = "#185FA5"
GREEN     = "#1D9E75"
RED       = "#E05C5C"
AMBER     = "#F5A623"
GREY      = "#8A8F98"
BG        = "#0F1117"       # Streamlit dark background
CARD_BG   = "#1A1D27"
GRID_CLR  = "rgba(255,255,255,0.06)"

SIGNAL_COLOURS = {"buy": GREEN, "hold": AMBER, "sell": RED}

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E0E0E0", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=True, gridcolor=GRID_CLR, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=GRID_CLR, zeroline=False),
)


def apply_defaults(fig: go.Figure) -> go.Figure:
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig


# ── 1. Sentiment heatmap ───────────────────────────────────────────────────────
def sentiment_heatmap(daily_df: pd.DataFrame) -> go.Figure:
    """
    Grid of tickers × dates coloured by daily sentiment score.
    Green = bullish, Red = bearish, Grey = neutral / no data.

    Args:
        daily_df: DataFrame with columns [ticker, date, sentiment_score]
    """
    pivot = (
        daily_df
        .pivot_table(index="ticker", columns="date", values="sentiment_score")
        .sort_index()
    )

    # Keep only last 30 trading days for readability
    pivot = pivot.iloc[:, -30:]

    # Format column labels as "Mon DD"
    col_labels = [
        pd.to_datetime(c).strftime("%b %d") for c in pivot.columns
    ]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=col_labels,
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  RED],
            [0.45, "rgba(80,80,80,0.4)"],
            [0.55, "rgba(80,80,80,0.4)"],
            [1.0,  GREEN],
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=pivot.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(
            title="Sentiment",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["Bearish", "", "Neutral", "", "Bullish"],
            thickness=12,
            len=0.8,
        ),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Score: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E0E0E0", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Sentiment Heatmap — Last 30 Trading Days", x=0.02),
        xaxis=dict(side="bottom", tickangle=-45, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        height=max(300, len(pivot.index) * 22 + 80),
    )
    return fig


# ── 2. Signal feed table (styled) ─────────────────────────────────────────────
def signal_table(signals_df: pd.DataFrame) -> go.Figure:
    """
    Colour-coded table of current buy/hold/sell signals.
    """
    if signals_df.empty:
        return go.Figure()

    df = signals_df.copy()

    # Emoji + label
    icon_map = {"buy": "▲ BUY", "sell": "▼ SELL", "hold": "● HOLD"}
    df["Signal"]     = df["signal"].map(icon_map)
    df["Ticker"]     = df["ticker"]
    df["Confidence"] = (df["confidence"] * 100).round(1).astype(str) + "%"
    df["Sentiment"]  = df["sentiment_score"].apply(lambda x: f"{x:+.3f}")
    df["RSI"]        = df["rsi_14"].round(1)
    df["5d Return"]  = (df["return_5d"] * 100).round(2).astype(str) + "%"

    display_cols = ["Ticker", "Signal", "Confidence", "Sentiment", "RSI", "5d Return"]
    cell_vals    = [df[c].tolist() for c in display_cols]

    # Cell colours based on signal
    signal_bg = df["signal"].map({
        "buy":  "rgba(29,158,117,0.18)",
        "sell": "rgba(224,92,92,0.18)",
        "hold": "rgba(245,166,35,0.12)",
    }).tolist()
    cell_colours = [["rgba(0,0,0,0)"] * len(df)] * (len(display_cols) - 1) + [signal_bg]
    # Colour the Signal column
    cell_colours[1] = signal_bg

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in display_cols],
            fill_color=CARD_BG,
            font=dict(color="#E0E0E0", size=12),
            align="left",
            line_color=GRID_CLR,
            height=32,
        ),
        cells=dict(
            values=cell_vals,
            fill_color=cell_colours,
            font=dict(color="#E0E0E0", size=11),
            align="left",
            line_color=GRID_CLR,
            height=28,
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E0E0E0", size=12),
        title=dict(text="Live Signal Feed", x=0.02),
        height=min(600, 80 + len(df) * 30),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ── 3. Equity curve ────────────────────────────────────────────────────────────
def equity_curve(results: dict) -> go.Figure:
    """
    Dual-panel chart: equity curve (top) + drawdown (bottom).
    Loaded from backtest_results.json.
    """
    # Reconstruct equity curves from trade log
    # (backtest.py stores summary metrics, not full daily series,
    #  so we re-simulate a simplified version for display)
    trade_log = results.get("trade_log", [])

    if not trade_log:
        fig = go.Figure()
        fig.add_annotation(text="Run backtest.py to generate results",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=GREY))
        return apply_defaults(fig)

    # Build a simple cumulative returns series from trade log
    trades = pd.DataFrame(trade_log)
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades = trades.sort_values("entry_date")

    daily = trades.groupby("entry_date")["return"].mean().fillna(0)
    equity_s = (1 + daily).cumprod()

    # Metrics for annotation
    strat = results.get("strategy", {})
    bench = results.get("benchmark", {})

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )

    # Equity line
    fig.add_trace(go.Scatter(
        x=equity_s.index,
        y=equity_s.values,
        mode="lines",
        name="FinSentinel",
        line=dict(color=BLUE, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(24,95,165,0.08)",
    ), row=1, col=1)

    # Baseline at 1.0
    fig.add_hline(y=1.0, line_dash="dot", line_color=GREY,
                  line_width=1, row=1, col=1)

    # Drawdown
    rolling_max = equity_s.cummax()
    drawdown    = (equity_s - rolling_max) / rolling_max

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        mode="lines",
        name="Drawdown",
        line=dict(color=RED, width=1),
        fill="tozeroy",
        fillcolor="rgba(224,92,92,0.25)",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=(
                f"Backtest Equity Curve  |  "
                f"Sharpe: {strat.get('sharpe_ratio', 0):.2f}  |  "
                f"Return: {strat.get('total_return', 0):+.1%}  |  "
                f"MaxDD: {strat.get('max_drawdown', 0):.1%}"
            ),
            x=0.02,
        ),
        height=400,
        legend=dict(x=0.01, y=0.97),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    return fig


# ── 4. Candlestick + sentiment overlay ────────────────────────────────────────
def price_sentiment_chart(
    price_df:     pd.DataFrame,
    sentiment_df: pd.DataFrame,
    ticker:       str,
    days:         int = 90,
) -> go.Figure:
    """
    Candlestick chart for a single ticker overlaid with
    a sentiment score line on a secondary y-axis.
    """
    price_df     = price_df[price_df["ticker"] == ticker].copy()
    sentiment_df = sentiment_df[sentiment_df["ticker"] == ticker].copy()

    # Limit to last N days
    price_df     = price_df.sort_values("date").tail(days)
    sentiment_df = sentiment_df[
        sentiment_df["date"] >= price_df["date"].min()
    ]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.04,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=price_df["date"],
        open=price_df["open"],
        high=price_df["high"],
        low=price_df["low"],
        close=price_df["close"],
        name="Price",
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        increasing_fillcolor=GREEN,
        decreasing_fillcolor=RED,
    ), row=1, col=1)

    # Volume bars
    vol_colours = [
        GREEN if c >= o else RED
        for c, o in zip(price_df["close"], price_df["open"])
    ]
    fig.add_trace(go.Bar(
        x=price_df["date"],
        y=price_df["volume"],
        name="Volume",
        marker_color=vol_colours,
        marker_opacity=0.5,
        showlegend=False,
    ), row=2, col=1)

    # Sentiment line overlay on price panel
    if not sentiment_df.empty:
        # Normalise sentiment to price range for overlay
        price_range  = price_df["close"].max() - price_df["close"].min()
        price_mid    = price_df["close"].mean()
        sent_norm    = (
            sentiment_df["sentiment_score"] * price_range * 0.3 + price_mid
        )

        fig.add_trace(go.Scatter(
            x=sentiment_df["date"],
            y=sent_norm,
            mode="lines",
            name="Sentiment",
            line=dict(color=AMBER, width=1.5, dash="dot"),
            opacity=0.8,
        ), row=1, col=1)

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"{ticker} — Price & Sentiment  (last {days}d)", x=0.02),
        height=480,
        xaxis_rangeslider_visible=False,
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


# ── 5. Sentiment distribution bar chart ───────────────────────────────────────
def sentiment_distribution(daily_df: pd.DataFrame) -> go.Figure:
    """
    Stacked bar chart showing bullish/neutral/bearish article counts per ticker.
    """
    needed = {"ticker", "bullish_count", "neutral_count", "bearish_count"}
    if not needed.issubset(daily_df.columns):
        return go.Figure()

    agg = (
        daily_df
        .groupby("ticker")[["bullish_count", "neutral_count", "bearish_count"]]
        .sum()
        .sort_values("bullish_count", ascending=True)
        .tail(20)
    )

    fig = go.Figure()
    for col, colour, label in [
        ("bullish_count",  GREEN, "Bullish"),
        ("neutral_count",  GREY,  "Neutral"),
        ("bearish_count",  RED,   "Bearish"),
    ]:
        fig.add_trace(go.Bar(
            x=agg[col],
            y=agg.index,
            orientation="h",
            name=label,
            marker_color=colour,
            marker_opacity=0.85,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E0E0E0", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Article Sentiment Breakdown by Ticker", x=0.02),
        barmode="stack",
        height=420,
        legend=dict(x=0.75, y=0.02),
        xaxis=dict(title="Article Count", gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    )
    return fig


# ── 6. Feature importance bar chart ───────────────────────────────────────────
def feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Horizontal bar chart of SHAP feature importances.
    importance_df must have columns: feature, importance
    """
    top = importance_df.head(top_n).sort_values("importance")

    colours = [
        GREEN if "sentiment" in f.lower() else
        BLUE  if any(k in f.lower() for k in ["rsi", "macd", "bb", "return", "volume"]) else
        AMBER
        for f in top["feature"]
    ]

    fig = go.Figure(go.Bar(
        x=top["importance"],
        y=top["feature"],
        orientation="h",
        marker_color=colours,
        marker_opacity=0.85,
        text=top["importance"].round(4),
        textposition="outside",
        textfont=dict(size=9),
    ))

    # Legend annotation
    fig.add_annotation(
        x=0.98, y=0.02, xref="paper", yref="paper",
        text="<span style='color:#1D9E75'>■</span> Sentiment  "
             "<span style='color:#185FA5'>■</span> Technical  "
             "<span style='color:#F5A623'>■</span> Other",
        showarrow=False, font=dict(size=10), align="right",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E0E0E0", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=f"Top {top_n} Features — Mean |SHAP Value|", x=0.02),
        height=max(300, top_n * 26 + 80),
        xaxis=dict(title="Mean |SHAP Value|", gridcolor=GRID_CLR, zeroline=False),
        yaxis=dict(gridcolor=GRID_CLR, zeroline=False),
    )
    return fig


# ── 7. Metric cards (rendered as tiny bar charts) ─────────────────────────────
def metric_sparkline(values: list[float], label: str, colour: str = BLUE) -> go.Figure:
    """
    Mini sparkline chart for metric cards on the overview page.
    """
    fig = go.Figure(go.Scatter(
        y=values,
        mode="lines",
        line=dict(color=colour, width=2),
        fill="tozeroy",
        fillcolor=colour.replace(")", ", 0.15)").replace("rgb", "rgba")
        if "rgb" in colour else colour + "26",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=60,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig