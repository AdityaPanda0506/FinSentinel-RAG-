"""
backtest.py
-----------
Backtests the trained signal model over historical data and computes
professional-grade quantitative performance metrics.

Strategy:
    - On each trading day, take the model's signal for each ticker
    - BUY  → go long, hold for 3 days
    - SELL → skip (no shorting for simplicity; can be extended)
    - HOLD → stay in cash
    - Equal-weight portfolio across all active BUY signals on a given day
    - Compare against simple buy-and-hold SPY benchmark

Metrics reported:
    - Total return
    - Annualised return
    - Sharpe ratio        (key metric — risk-adjusted return)
    - Sortino ratio       (like Sharpe but only penalises downside volatility)
    - Maximum drawdown    (worst peak-to-trough loss)
    - Win rate            (% of trades that were profitable)
    - Average trade return
    - Calmar ratio        (annualised return / max drawdown)

Usage:
    python ml/backtest.py
    python ml/backtest.py --ticker AAPL    # backtest on single ticker
    python ml/backtest.py --plot           # save equity curve chart

Output:
    data/backtest_results.json    — metrics dict
    models/equity_curve.png       — equity curve chart (with --plot)
"""

import json
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.train import load_artifacts

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
MODELS_DIR   = ROOT_DIR / "models"
FEATURES_CSV = DATA_DIR / "features.csv"
RESULTS_JSON = DATA_DIR / "backtest_results.json"
EQUITY_PLOT  = MODELS_DIR / "equity_curve.png"

# ── Config ─────────────────────────────────────────────────────────────────────
HOLD_DAYS       = 3       # hold each position for N trading days
RISK_FREE_RATE  = 0.05    # annual risk-free rate (5% = approx 2024 US T-bills)
TRADING_DAYS    = 252     # trading days per year

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Metrics ────────────────────────────────────────────────────────────────────
def sharpe_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Annualised Sharpe ratio."""
    if returns.std() == 0:
        return 0.0
    excess = returns - (rf / TRADING_DAYS)
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """
    Like Sharpe but only penalises downside volatility.
    Better metric for strategies that avoid large losses.
    """
    excess      = returns - (rf / TRADING_DAYS)
    downside    = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough decline."""
    rolling_max = equity_curve.cummax()
    drawdown    = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def calmar_ratio(annual_return: float, mdd: float) -> float:
    """Annual return divided by absolute max drawdown."""
    if mdd == 0:
        return 0.0
    return annual_return / abs(mdd)


def annualised_return(total_return: float, n_days: int) -> float:
    """Compound annual growth rate."""
    if n_days == 0:
        return 0.0
    return float((1 + total_return) ** (TRADING_DAYS / n_days) - 1)


# ── Backtest engine ────────────────────────────────────────────────────────────
def run_backtest(
    df:       pd.DataFrame,
    feat_cols: list[str],
    model,
    le,
    test_start_ratio: float = 0.85,  # only test on last 15% of data (out-of-sample)
) -> tuple[pd.Series, list[dict]]:
    """
    Run the strategy on out-of-sample test data.

    Returns:
        strategy_returns  : daily portfolio returns (pd.Series indexed by date)
        trade_log         : list of individual trade dicts
    """
    # Use only the out-of-sample portion
    dates     = df["date"].sort_values().unique()
    cutoff    = dates[int(len(dates) * test_start_ratio)]
    test_df   = df[df["date"] >= cutoff].copy()

    log.info("Backtesting on %d rows from %s → %s",
             len(test_df),
             test_df["date"].min().date(),
             test_df["date"].max().date())

    test_df["signal_pred"] = le.inverse_transform(
        model.predict(test_df[feat_cols].fillna(0))
    )

    # Build a price lookup: ticker → {date: close}
    price_lookup = (
        test_df.set_index(["ticker", "date"])["close"]
        .to_dict()
    )

    daily_returns = {}
    trade_log     = []

    all_dates = sorted(test_df["date"].unique())

    for i, entry_date in enumerate(all_dates):
        day_df = test_df[test_df["date"] == entry_date]

        # Get buy signals for today
        buy_signals = day_df[day_df["signal_pred"] == "buy"]

        if buy_signals.empty:
            daily_returns[entry_date] = 0.0
            continue

        # Equal-weight across all buy signals
        trade_returns = []

        for _, row in buy_signals.iterrows():
            ticker       = row["ticker"]
            entry_price  = row["close"]

            # Exit after HOLD_DAYS
            exit_idx   = i + HOLD_DAYS
            if exit_idx >= len(all_dates):
                continue  # not enough future data

            exit_date  = all_dates[exit_idx]
            exit_price = price_lookup.get((ticker, exit_date))

            if exit_price is None or entry_price == 0:
                continue

            trade_ret = (exit_price - entry_price) / entry_price
            trade_returns.append(trade_ret)

            trade_log.append({
                "ticker":       ticker,
                "entry_date":   str(entry_date.date()),
                "exit_date":    str(exit_date.date()),
                "entry_price":  round(entry_price, 2),
                "exit_price":   round(exit_price, 2),
                "return":       round(trade_ret, 4),
                "profitable":   trade_ret > 0,
            })

        # Portfolio return = average of all trades entered today
        daily_returns[entry_date] = (
            float(np.mean(trade_returns)) if trade_returns else 0.0
        )

    returns_series = pd.Series(daily_returns).sort_index()
    return returns_series, trade_log


def benchmark_buy_and_hold(df: pd.DataFrame, test_start_ratio: float = 0.85) -> pd.Series:
    """
    Simple equal-weight buy-and-hold of all tickers in the test period.
    This is the baseline we compare against.
    """
    dates   = df["date"].sort_values().unique()
    cutoff  = dates[int(len(dates) * test_start_ratio)]
    test_df = df[df["date"] >= cutoff]

    bnh = (
        test_df.groupby("date")["close"]
        .mean()
        .pct_change()
        .fillna(0)
    )
    return bnh


# ── Metrics summary ────────────────────────────────────────────────────────────
def compute_metrics(
    returns:    pd.Series,
    trade_log:  list[dict],
    label:      str = "strategy",
) -> dict:
    equity     = (1 + returns).cumprod()
    total_ret  = float(equity.iloc[-1] - 1) if not equity.empty else 0.0
    n_days     = len(returns)
    annual_ret = annualised_return(total_ret, n_days)
    mdd        = max_drawdown(equity)

    winning_trades  = [t for t in trade_log if t.get("profitable")]
    total_trades    = len(trade_log)
    win_rate        = len(winning_trades) / total_trades if total_trades > 0 else 0
    avg_trade_ret   = np.mean([t["return"] for t in trade_log]) if trade_log else 0

    metrics = {
        "label":                label,
        "total_return":         round(total_ret,   4),
        "annualised_return":    round(annual_ret,  4),
        "sharpe_ratio":         round(sharpe_ratio(returns),  3),
        "sortino_ratio":        round(sortino_ratio(returns), 3),
        "max_drawdown":         round(mdd,          4),
        "calmar_ratio":         round(calmar_ratio(annual_ret, mdd), 3),
        "win_rate":             round(win_rate,     4),
        "total_trades":         total_trades,
        "avg_trade_return":     round(float(avg_trade_ret), 4),
        "n_trading_days":       n_days,
    }
    return metrics


def print_metrics(m: dict) -> None:
    log.info("─" * 60)
    log.info("%-28s  %s", "Metric", m["label"].upper())
    log.info("─" * 60)
    log.info("%-28s  %+.1f%%",  "Total Return",       m["total_return"]      * 100)
    log.info("%-28s  %+.1f%%",  "Annualised Return",  m["annualised_return"] * 100)
    log.info("%-28s  %.3f",     "Sharpe Ratio",       m["sharpe_ratio"])
    log.info("%-28s  %.3f",     "Sortino Ratio",      m["sortino_ratio"])
    log.info("%-28s  %.1f%%",   "Max Drawdown",       m["max_drawdown"]      * 100)
    log.info("%-28s  %.3f",     "Calmar Ratio",       m["calmar_ratio"])
    log.info("%-28s  %.1f%%",   "Win Rate",           m["win_rate"]          * 100)
    log.info("%-28s  %d",       "Total Trades",       m["total_trades"])
    log.info("%-28s  %+.2f%%",  "Avg Trade Return",   m["avg_trade_return"]  * 100)


# ── Equity curve plot ──────────────────────────────────────────────────────────
def plot_equity_curve(
    strategy_returns:  pd.Series,
    benchmark_returns: pd.Series,
    strategy_metrics:  dict,
    save_path:         Path = EQUITY_PLOT,
) -> None:
    strat_equity = (1 + strategy_returns).cumprod()
    bench_equity = (1 + benchmark_returns).cumprod()

    # Align both series to same dates
    combined = pd.DataFrame({
        "FinSentinel Strategy": strat_equity,
        "Buy & Hold Benchmark": bench_equity,
    }).dropna()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                             gridspec_kw={"height_ratios": [3, 1]})

    # ── Top panel: equity curves ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(combined.index, combined["FinSentinel Strategy"],
            color="#185FA5", linewidth=2, label="FinSentinel Strategy")
    ax.plot(combined.index, combined["Buy & Hold Benchmark"],
            color="#888888", linewidth=1.5, linestyle="--", label="Buy & Hold")
    ax.fill_between(combined.index, combined["FinSentinel Strategy"], 1,
                    alpha=0.08, color="#185FA5")

    ax.set_title(
        f"FinSentinel Backtest  |  "
        f"Sharpe: {strategy_metrics['sharpe_ratio']:.2f}  |  "
        f"Return: {strategy_metrics['total_return']:+.1%}  |  "
        f"MaxDD: {strategy_metrics['max_drawdown']:.1%}",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Portfolio Value (starting = 1.0)", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")

    # ── Bottom panel: drawdown ────────────────────────────────────────────────
    ax2 = axes[1]
    rolling_max = strat_equity.cummax()
    drawdown    = (strat_equity - rolling_max) / rolling_max

    ax2.fill_between(drawdown.index, drawdown, 0,
                     color="#E05C5C", alpha=0.6)
    ax2.set_ylabel("Drawdown", fontsize=10)
    ax2.set_ylim(min(drawdown.min() * 1.2, -0.01), 0.01)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0%}")
    )
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Equity curve saved → %s", save_path)


# ── Main ───────────────────────────────────────────────────────────────────────
def run(
    ticker_filter: str | None = None,
    save_plot:     bool = True,
) -> dict:
    if not FEATURES_CSV.exists():
        log.error("features.csv not found. Run feature_engineering.py first.")
        return {}

    model, le, feat_cols = load_artifacts()

    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    if ticker_filter:
        df = df[df["ticker"] == ticker_filter.upper()]

    if df.empty or len(df) < 100:
        log.error("Not enough data to backtest.")
        return {}

    # Run strategy
    strategy_returns, trade_log = run_backtest(df, feat_cols, model, le)

    # Run benchmark
    benchmark_returns = benchmark_buy_and_hold(df)
    benchmark_returns = benchmark_returns.reindex(strategy_returns.index).fillna(0)

    # Compute metrics
    strat_metrics = compute_metrics(strategy_returns, trade_log, label="FinSentinel")
    bench_metrics = compute_metrics(benchmark_returns, [], label="Buy & Hold")

    print_metrics(strat_metrics)
    print_metrics(bench_metrics)

    # Alpha (outperformance vs benchmark)
    alpha = strat_metrics["annualised_return"] - bench_metrics["annualised_return"]
    log.info("─" * 60)
    log.info("Alpha vs buy-and-hold: %+.1f%%", alpha * 100)

    # Save results
    results = {
        "strategy":  strat_metrics,
        "benchmark": bench_metrics,
        "alpha":     round(alpha, 4),
        "trade_log": trade_log[:50],   # save first 50 trades as sample
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved → %s", RESULTS_JSON)

    # Plot
    if save_plot:
        plot_equity_curve(strategy_returns, benchmark_returns, strat_metrics)

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the signal model")
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--plot",   action="store_true",
                        help="Save equity curve chart")
    args = parser.parse_args()
    run(ticker_filter=args.ticker, save_plot=args.plot)