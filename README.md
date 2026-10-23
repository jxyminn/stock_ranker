# Stock Ranker

Ranks stocks for 5/10/15/30-day horizons using a composite score built from:
- a **regression model's** predicted forward return
- a **classification model's** predicted probability the price is up
- a **risk-adjusted** score (predicted return ÷ recent volatility)

Models are gradient-boosted trees (LightGBM), one regressor + one classifier
per horizon, trained on technical-indicator features (RSI, MACD, moving
averages, Bollinger Bands, volume ratios, etc.), earnings-calendar features
(days since/until the nearest earnings report, last surprise %), and
market-regime features (S&P 500 / VIX), with no lookahead.

## Setup

```bash
cd stock_ranker
pip install -r requirements.txt
```

There's also a Streamlit UI if you'd rather not use the CLI:

```bash
streamlit run app.py
```

`app.py` opens on **Stock Research** — search one stock and see model
predictions, technicals, earnings timing, market regime, fundamentals, and
news, all in one place. The multi-stock **Ranker** (rank many stocks against
each other) is available as a secondary page via the sidebar page picker.

## 1. Train the models (run this first, and periodically re-run to keep models current)

```bash
python train.py
```

This trains on a built-in list of ~50 large, liquid US stocks (see
`config.DEFAULT_TRAIN_UNIVERSE`) using ~5 years of history. You can train on
your own universe instead:

```bash
python train.py --universe AAPL MSFT NVDA TSLA AMD ...
```

Models are saved to `models/horizon_{5,10,15,30}.joblib`. Each training run
prints a holdout MAE (regression error) and AUC (classification skill;
0.5 = no better than a coin flip, so treat models near 0.5 with real
skepticism).

Each saved model also records the exact feature list it was trained on. If
you (or a future version of this code) change `features.py` and forget to
retrain, `predict.load_models()` will fail loudly with a clear "retrain
needed" message instead of a cryptic LightGBM shape-mismatch error.

## 2. Search for a ticker (optional)

If you don't know the exact symbol:

```bash
python search.py "palantir"
```

## 3. Rank stocks with live data

```bash
python predict.py --tickers AAPL MSFT NVDA AMD PLTR
# or
python predict.py --tickers-file my_watchlist.txt --top 20
```

This pulls the latest live price/volume data for your list, computes the same
features used in training, and prints a ranked table per horizon.

## 4. Deep-dive on one stock (optional)

For a single-stock "should I invest" report — model predictions per horizon
(with each model's own reliability), a plain-language technical read,
earnings timing, market regime, company fundamentals (P/E, market cap,
sector, analyst targets), and recent headlines with sentiment scoring:

```bash
python analyze.py AAPL
python analyze.py "palantir"   # company name works too
```

The same report is available in the Streamlit UI — it's the main **Stock
Research** page (`streamlit run app.py`), which also adds a price chart
(close price, 20d/50d moving averages, Bollinger Bands, volume, and
earnings-date markers) that the CLI report doesn't show.

Fundamentals and headlines/sentiment are pulled fresh for this report only —
they are *not* seen by the trained models (see caveats below). Sentiment
scoring uses VADER extended with a small finance-specific vocabulary (see
`sentiment.py`) — it's a heuristic, not a trained model; read the headlines
yourself too.

## 5. Check whether the ranking is actually any good (recommended)

`train.py`'s printed MAE/AUC comes from a single train/test split, which is a
noisy read on model quality. For a fuller picture:

```bash
python backtest.py
```

This walks forward through time in several windows, retraining on an
expanding window each time (no lookahead) and scoring the live cross-section
exactly like `predict.py` would. It reports the Information Coefficient (IC —
rank correlation between predicted scores and what actually happened) and a
top-vs-bottom spread per fold, which is a much more honest signal than a
single holdout AUC. Run this after every retrain before trusting the live
rankings.

Each run also saves a summary to `models/backtest_results.json`, which
`predict.py`, `analyze.py`, and the Streamlit app read to show a **horizon
reliability ranking** ("30-day picks have been the most reliable, 5-day the
least") — so you know which horizon's predictions to actually lean on. This
is separate from the per-horizon MAE/AUC shown elsewhere, which comes from
`train.py`'s single noisy holdout split; the ranking is only as fresh as your
last `backtest.py` run.

To tune the LightGBM hyperparameters against that same walk-forward metric
instead of guessing:

```bash
python tune.py
```

It runs a small grid search and reports the best config by mean IC; update
`config.LGBM_PARAMS` with whatever it finds and re-run `train.py`.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests run offline (network calls are mocked) and cover the no-lookahead
guarantees, feature engineering edge cases, caching, and the walk-forward
backtest logic. Several of these tests exist because they caught real bugs
during development (a timezone-mismatch bug that silently zeroed out VIX
data, a cache key collision that silently truncated training history, an RSI
edge case that produced NaN instead of 100, and a duplicate-index bug that
silently zeroed the backtest's spread metric) — run them after changing
`features.py`, `data_fetcher.py`, or `backtest.py`.

## Re-running for fresh data

`predict.py` always fetches live data (no caching), so every run reflects the
latest available prices. The **models** themselves are static once trained —
re-run `train.py` periodically (e.g. weekly) so they don't go stale as market
regimes shift. You can put `train.py` on a cron job / scheduled task.

## Nightly briefing

`nightly_briefing.py` pulls fresh data for every ticker in
`config.WATCHLIST_SECTIONS`, runs the already-trained models for a forecast,
and compares the result to the previous run to flag what's actually *new*:
a forecast or chance-of-gain swing beyond the thresholds in `config.py`, an
earnings report due within a week, a golden/death cross that just happened,
a new near-all-time-high, or notably one-sided news. It does **not** retrain
the models — that's still a separate, manual, much slower step (`train.py`).

In this deployment it runs automatically every night via
`.github/workflows/nightly.yml` (a GitHub Actions scheduled job — no
computer needs to stay on), which commits `briefing/snapshot.json` (raw data,
for tomorrow's diff) and `briefing/latest.json` (the digest) back to the
repo. The **Daily Briefing** Streamlit page just reads `briefing/latest.json`
as a static file — it never calls yfinance itself, so it loads instantly.
You can also run it manually any time with `python nightly_briefing.py`, or
trigger the scheduled job early from the repo's Actions tab.

## Important caveats — please read

- **This is not a reliable way to beat the market.** Stock prices are close
  to a random walk over short horizons; a healthy holdout AUC on this kind of
  data is often only modestly above 0.5. Look at the printed MAE/AUC before
  trusting any ranking.
- The models see technical/price data, earnings *timing* (when the last/next
  report is and whether the last one beat/missed), and market-regime data
  (S&P 500 / VIX) — but not earnings *content*, news, or fundamentals
  (P/E, sector, analyst targets, etc.). `analyze.py`'s report shows
  fundamentals and news sentiment for human context, but the trained models
  never see them — there's no historical news archive available to backfill
  training data without lookahead, so it can only ever be a live, report-only
  signal. A stock can rank highly and still move against the prediction on
  any given day, especially around an earnings date, which is inherently
  higher-volatility and harder to predict.
- **Backtest before trusting live output.** Run `python backtest.py` for a
  proper walk-forward check (predict → wait → compare to actual) before
  relying on this for real decisions — don't trust `train.py`'s single
  holdout split alone.
- Not financial advice — treat this as a research/learning tool.

## File overview

| File | Purpose |
|---|---|
| `config.py` | Horizons, training universe, score weights, LightGBM hyperparameters |
| `data_fetcher.py` | Live/historical data pulls, ticker search, fundamentals (yfinance) |
| `features.py` | Technical, earnings-calendar, and market-regime feature engineering |
| `labeling.py` | Forward-return training labels (training only) |
| `train.py` | Trains + saves per-horizon models |
| `predict.py` | Scores/ranks a ticker list using live data |
| `backtest.py` | Walk-forward IC backtest — the real check on model quality |
| `reliability.py` | Saves/reads `backtest.py`'s results; ranks horizons by reliability |
| `tune.py` | Small hyperparameter grid search scored by walk-forward IC |
| `analyze.py` | Single-stock deep-dive report (CLI) |
| `sentiment.py` | Headline sentiment scoring (report-only, not a model input) |
| `insights.py` | Shared plain-English technical read-outs (used by `app.py` and `analyze.py`) |
| `search.py` | CLI ticker lookup by company name |
| `app.py` | Streamlit UI — single-stock research (main page) |
| `pages/1_Ranker.py` | Streamlit UI — multi-stock ranking (secondary page) |
| `pages/2_My_Watchlist.py` | Streamlit UI — per-stock info browser for `config.WATCHLIST_SECTIONS` |
| `pages/3_Daily_Briefing.py` | Streamlit UI — reads `briefing/latest.json`, shows what changed overnight |
| `nightly_briefing.py` | Generates the nightly watchlist digest (see "Nightly briefing" above) |
| `.github/workflows/nightly.yml` | Scheduled GitHub Actions job that runs `nightly_briefing.py` and commits the result |
| `tests/` | pytest suite — see "Running the tests" above |
