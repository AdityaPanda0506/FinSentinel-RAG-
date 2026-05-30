<div align="center">

# 📈 FinSentinel

### AI-Powered Market Sentiment Signal Engine

*From raw financial news to explainable trading signals — end to end.*

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)
[![FinBERT](https://img.shields.io/badge/NLP-FinBERT-6B46C1?style=for-the-badge)](https://huggingface.co/ProsusAI/finbert)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-F59E0B?style=for-the-badge)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

[**Report Bug**](issues) · [**Features**](#features) · [**Setup**](#setup)

---

> FinSentinel ingests 3M+ financial news articles, scores sentiment using FinBERT,
> engineers 42 features combining NLP signals with technical indicators and macro data,
> trains an XGBoost + LightGBM ensemble, and validates every prediction through a
> professional backtesting engine — deployed as an interactive Streamlit dashboard.

</div>

---

## 📸 Screenshots

> **Add these screenshots to your repo — instructions at the bottom of this README**

| Page | Screenshot |
|------|-----------|
| Overview — Sentiment Heatmap | `docs/screenshots/01_overview.png` |
| Live Signals — Signal Feed | `docs/screenshots/02_signals.png` |
| Ticker Deep Dive — AAPL | `docs/screenshots/03_deepdive.png` |
| Backtest — Equity Curve | `docs/screenshots/04_backtest.png` |
| Explainability — SHAP | `docs/screenshots/05_shap.png` |

<img width="1777" height="862" alt="01_overview" src="https://github.com/user-attachments/assets/24d5f667-16be-4c12-8092-c73d144f0c3e" />
<img width="1533" height="842" alt="02_signals" src="https://github.com/user-attachments/assets/bf99c9e5-9f0a-497f-a7fd-edad7f24d84e" />
<img width="1526" height="818" alt="03_deepdive" src="https://github.com/user-attachments/assets/3f6d9181-4b79-4f27-98e0-a50d381d9ecd" />
<img width="1208" height="799" alt="04_backtest" src="https://github.com/user-attachments/assets/e90d58e7-6b2b-4610-a718-be452063dc47" />
<img width="1481" height="881" alt="05_shap_summary" src="https://github.com/user-attachments/assets/cc1b6064-7b4e-4b5b-90e7-579e81538ff1" />

---

## ✨ Features

- **Multi-source ingestion** — NewsAPI, Kaggle (3M+ articles), Stocktwits, Yahoo Finance across 100 tickers
- **FinBERT sentiment scoring** — finance-domain BERT model, not generic sentiment — understands "beat expectations" vs "missed by a wide margin"
- **42-feature engineering** — price momentum, RSI, MACD, Bollinger Bands, VIX, 10Y yield, Gold, sentiment lags, earnings proximity, sector encoding
- **XGBoost + LightGBM ensemble** — soft-voting ensemble with walk-forward validation to prevent lookahead bias
- **Professional backtesting** — Sharpe, Sortino, max drawdown, Calmar ratio, equity curve vs buy-and-hold
- **SHAP explainability** — every signal comes with a quantified explanation of which features drove it
- **Interactive 5-page dashboard** — heatmap, signal feed, ticker deep-dive, backtest results, explainability

---

## 🏗️ System Architecture

<img width="2752" height="1536" alt="architecture" src="https://github.com/user-attachments/assets/46120cc6-f164-4fee-8b9b-f88e9055fb08" />


---

## 📊 Results

### Model Performance (Out-of-Sample)

| Metric | Value | Context |
|--------|-------|---------|
| Overall Accuracy | **54.2%** | vs 33% random baseline (3-class) |
| Directional Accuracy | **29.2%** | buy/sell predictions only |
| Win Rate | **50.8%** | % of trades that were profitable |
| Test Period | 2018–2020 | Strict out-of-sample |
| Test Samples | 9,179 | Across 39 S&P 500 tickers |
| Total Trades Backtested | 9,678 | Long positions only |

### Strategy vs Benchmark

| Metric | FinSentinel | Buy & Hold |
|--------|-------------|------------|
| Total Return | -2.8% | +11.1% |
| Annualised Return | -3.6% | +15.0% |
| Sharpe Ratio | -0.17 | 0.78 |
| Sortino Ratio | -0.21 | 1.07 |
| Max Drawdown | -26.8% | -6.8% |

> The strategy underperformed buy-and-hold in the 2024 bull market — expected for a conservative signal model. The model generates HOLD on uncertain days rather than buying blindly, which costs return in strong uptrends but reduces drawdown risk in corrections.

### Top 10 Predictive Features (SHAP)

```
Rank  Feature                Description
────  ─────────────────────  ─────────────────────────────────────
 1    hl_range               Intraday high-low volatility ratio
 2    tnx_yield              10-year US Treasury yield
 3    vix                    CBOE Volatility Index (fear gauge)
 4    spy_return             S&P 500 daily return (market direction)
 5    sector_encoded         GICS sector of the stock
 6    bb_width               Bollinger Band width (volatility proxy)
 7    tlt_return             20-year Treasury ETF return
 8    gld_return             Gold ETF return (risk-off signal)
 9    rsi_14                 14-day Relative Strength Index
10    sentiment_score        FinBERT daily confidence-weighted score
```

**Key insight:** Macro features (VIX, Treasury yield, SPY) dominate over individual sentiment — consistent with academic research showing macro conditions explain ~60% of individual stock variance.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| NLP Model | FinBERT (ProsusAI) | Finance-domain sentiment classification |
| ML | XGBoost + LightGBM | Signal classification ensemble |
| Explainability | SHAP | Feature attribution per prediction |
| Prices | yfinance | Free OHLCV data, 100+ tickers |
| News | NewsAPI + Kaggle | 3M+ historical + live headlines |
| Storage | SQLite + CSV | Lightweight, portable, no server needed |
| Dashboard | Streamlit + Plotly | Interactive web app |
| NLP utils | spaCy + HuggingFace | Entity extraction + tokenisation |
| Language | Python 3.12 | — |

---

## 📁 Project Structure

```
finsentinel/
│
├── 📂 ingestion/                   Data collection
│   ├── price_fetcher.py            Yahoo Finance OHLCV → SQLite
│   ├── news_scraper.py             NewsAPI + Kaggle CSV → SQLite
│   ├── reddit_scraper.py           Reddit finance posts (PRAW)
│   ├── stocktwits_scraper.py       Retail sentiment, no auth needed
│   └── tickers.json                100+ ticker watchlist + sectors
│
├── 📂 nlp/                         Sentiment pipeline
│   ├── sentiment_pipeline.py       FinBERT batched inference
│   ├── aggregator.py               Daily confidence-weighted scores
│   └── entity_extractor.py         Ticker tagging (regex + spaCy NER)
│
├── 📂 ml/                          Machine learning
│   ├── feature_engineering.py      42-feature matrix builder
│   ├── train.py                    XGBoost + LightGBM ensemble
│   ├── predict.py                  Live signal generation
│   ├── explain.py                  SHAP analysis + plots
│   └── backtest.py                 Quant metrics + equity curve
│
├── 📂 dashboard/                   Streamlit app
│   ├── app.py                      Entry point (5 pages)
│   ├── charts.py                   Plotly chart builders
│   ├── components.py               Reusable UI components + data loaders
│   └── ticker_meta.py              Full names, sectors, emojis
│
├── 📂 data/                        Generated — gitignored
│   ├── raw_prices.db               SQLite price history
│   ├── raw_news.db                 SQLite 3M+ articles
│   ├── sentiment_scores.csv        Per-headline FinBERT output
│   ├── daily_sentiment.csv         Aggregated daily per ticker
│   ├── features.csv                42-feature matrix
│   ├── latest_signals.csv          Current signals
│   └── backtest_results.json       Metrics + trade log
│
├── 📂 models/                      Trained artifacts — gitignored
│   ├── xgb_signal_model.pkl        Ensemble model
│   ├── label_encoder.pkl           Signal encoder
│   ├── feature_columns.json        Feature list for inference
│   ├── shap_summary.png            Global importance chart
│   └── equity_curve.png            Backtest equity curve
│
├── 📂 docs/                        Documentation assets
│   └── screenshots/                Dashboard screenshots (see below)
│
├── config.py                       Centralised config (loads .env)
├── run_pipeline.py                 Daily pipeline orchestrator
├── get_metrics.py                  Performance report generator
├── requirements.txt                All dependencies pinned
├── .env.example                    API key template
├── .gitignore                      Excludes data/, models/, .env
└── README.md                       This file
```

---

## ⚡ Setup

### Requirements

- Python 3.10+
- 4GB RAM (8GB recommended for FinBERT)
- 5GB disk space

### Step 1 — Clone and install

```bash
git clone https://github.com/yourusername/finsentinel.git
cd finsentinel
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 2 — Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:
```
NEWSAPI_KEY=your_key_here        # Free at newsapi.org/register
REDDIT_CLIENT_ID=optional
REDDIT_CLIENT_SECRET=optional
REDDIT_USER_AGENT=finsentinel_v1
```

### Step 3 — Download Kaggle datasets

Download and place in `data/`:
- [Massive Stock News Analysis](https://kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests)
- [Financial News Sentiment](https://kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news)

---

## 🚀 Usage

### First-time full pipeline

```bash
# 1. Fetch price history (100+ tickers, 2009–2024)
python ingestion/price_fetcher.py

# 2. Load Kaggle historical news
python ingestion/news_scraper.py --kaggle data/analyst_ratings_processed.csv
python ingestion/news_scraper.py --kaggle data/raw_partner_headlines.csv

# 3. Scrape live news (last 30 days from NewsAPI)
python ingestion/news_scraper.py

# 4. Score sentiment with FinBERT (30–60 min on CPU)
# PowerShell:
$tickers = @("AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","JPM","BAC","GS")
foreach ($t in $tickers) { python nlp/sentiment_pipeline.py --ticker $t --source news }

# 5. Aggregate daily scores
python nlp/aggregator.py

# 6. Build feature matrix
python ml/feature_engineering.py

# 7. Train model
python ml/train.py

# 8. Generate signals + backtest
python ml/predict.py
python ml/backtest.py --plot
python ml/explain.py

# 9. Launch dashboard
streamlit run dashboard/app.py
```

### Daily update

```bash
python run_pipeline.py
```

### Windows — set PYTHONPATH first

```powershell
# Temporary (current session)
$env:PYTHONPATH = "C:\path\to\finsentinel"

# Permanent (all future sessions)
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", "C:\path\to\finsentinel", "User")
```

### View performance metrics

```bash
python get_metrics.py
```

---

## 🎯 Key Design Decisions

**Walk-forward validation — not random split**
Random train/test splits on time-series data constitute lookahead bias — the model sees future market conditions during training, producing metrics that collapse in production. Every iteration of FinSentinel uses strict temporal splits.

**FinBERT over VADER or TextBlob**
Generic sentiment models are trained on social media and product reviews. FinBERT is pre-trained on SEC filings, earnings call transcripts, and financial news. It correctly classifies domain-specific phrases like "margin compression" (bearish) and "beat on top and bottom line" (bullish) that generic models miss entirely.

**Confidence-weighted aggregation**
A headline FinBERT scores at 95% confidence counts 1.82x more than one scored at 52% confidence in the daily aggregation. This signal quality weighting produces a cleaner sentiment metric than simple averaging.

**Macro features as first-class inputs**
Individual stock sentiment explains maybe 10–15% of short-term price variance. The remaining 85–90% is driven by macro conditions — interest rates, market-wide risk appetite, sector rotation. VIX and Treasury yield being the top SHAP features confirms this and shows the model learned real market dynamics.

**SHAP for explainability**
Black-box predictions are a trust and regulatory concern in finance. SHAP provides mathematically rigorous Shapley values — every prediction is explained by exactly how much each feature contributed, in the same units as the model output.

---

## ⚠️ Limitations

**Sentiment data gap (2021–2024)**
The Kaggle dataset covers 2009–2020. Post-2020 sentiment relies on rolling NewsAPI collection (30-day window). This creates a distribution shift between training and recent test periods — diagnosed and documented as a research finding.

**Long-only backtest**
The current strategy only takes long positions on BUY signals. Adding short positions on SELL signals would significantly change the risk profile.

**Daily granularity**
FinBERT scores batch-processed at end-of-day. Intraday news can move stocks within minutes — a real-time pipeline (Alpaca API + WebSocket) would capture this signal before it decays.

**Free-tier API limits**
NewsAPI free tier: 100 requests/day, 30-day history. Upgrading to a paid tier would enable full historical news collection.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

## 👤 Author

**Aditya**

Built to demonstrate: NLP, time-series ML, quantitative finance, SHAP explainability, and production pipeline design.

---

*If this project helped you, consider giving it a ⭐ on GitHub*

</div>
