# EUR/USD Candle Colour Predictor

> Predicts whether the **next 15-minute EUR/USD candle will be 🟢 GREEN or 🔴 RED** using a full scikit-learn preprocessing pipeline and a soft-voting ensemble classifier, served via a FastAPI REST API with live yfinance data ingestion.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data](#data)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model](#model)
- [Performance](#performance)
- [API Reference](#api-reference)
- [Sample Requests](#sample-requests)
- [Testing](#testing)
- [Configuration](#configuration)
- [Workflow](#workflow)

---

## Overview

This project builds a complete, production-ready pipeline for predicting the **colour of the next 15-minute EUR/USD candle** (green = close > open, red = close ≤ open).

```
Raw OHLCV CSV  ──►  Preprocessing Pipeline  ──►  Ensemble Classifier  ──►  REST API
                    (cleaning + features +         (RF + HGB + LR             (FastAPI +
                     RobustScaler)                  soft-voting)               yfinance)
```

**What it does end-to-end:**

1. Downloads up to 365 days of 15-min OHLCV data from Yahoo Finance via the API
2. Cleans, validates and engineers 21 stationary features
3. Trains a calibrated soft-voting ensemble on a chronological 70/10/20 split
4. Serves live predictions via REST — a single `GET /predict/latest` call fetches fresh yfinance data and returns a prediction with no file upload required

---

## Project Structure

```
project/
│
├── run.py                          # Single entry-point (train / serve / all / predict)
├── pipeline.py                     # Master preprocessing pipeline + walk-forward split
├── train.py                        # Model training, evaluation, artifact saving
├── api.py                          # FastAPI application (all routes)
├── test_api.py                     # Full pytest test suite (57 tests)
├── requirements.txt
│
├── preprocessing/
│   ├── __init__.py
│   ├── cleaning.py                 # minimal_pipeline, enhanced_pipeline, gap/outlier helpers
│   └── feature_pipe.py             # FeatureEngineer, WarmupDropper, FeatureSelector,
│                                   # FeatureScaler, FEATURE_COLS registry
│
└── data/
    ├── raw/
    │   └── OANDA_EURUSD_15.csv     # Source OHLCV data (place yours here)
    ├── processed/
    │   ├── train_raw.pkl           # Raw train split (pre-feature-engineering)
    │   ├── val_raw.pkl
    │   └── test_raw.pkl
    └── artifacts/
        ├── master_pipe.pkl         # Fitted preprocessing pipeline
        ├── candle_colour_model.pkl # Fitted ensemble classifier
        ├── feature_scaler.pkl      # Fitted RobustScaler (standalone)
        └── metrics.json            # Val + test evaluation metrics
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Requirements: Python ≥ 3.10**

### 2. Add your raw data

Place your `OANDA_EURUSD_15.csv` in `data/raw/`. Expected columns:

```
time, open, high, low, close, Volume
```

Or skip this step and download directly via the API after starting the server (see [Download via API](#download-via-api)).

### 3. Train

```bash
python run.py train
```

This runs the full preprocessing pipeline, saves processed splits, then trains and evaluates the model. All artifacts are saved to `data/artifacts/`.

### 4. Serve

```bash
python run.py serve
```

API available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 5. Predict

```bash
# CLI — instant prediction from terminal
python run.py predict

# Or via curl
curl "http://localhost:8000/predict/latest"
```

### One-liner (train + serve)

```bash
python run.py all
```

---

## Data

| Property | Value |
|---|---|
| Instrument | EUR/USD |
| Timeframe | 15 minutes |
| Source | OANDA (historical) / Yahoo Finance `EURUSD=X` (live) |
| Raw rows | 20,512 candles |
| Date range | May 2025 → Feb 2026 |
| Columns | `time, open, high, low, close, Volume` |

### Splits (chronological — no shuffling)

| Split | Rows | Fraction |
|---|---|---|
| Train | 14,358 | 70% |
| Validation | 2,051 | 10% |
| Test | 4,103 | 20% |

### Download fresh data via the API

```bash
# Start server first, then:
curl -s -X POST "http://localhost:8000/data/download" | python3 -m json.tool

# Custom: GBPUSD, 1-hour bars, 2 years
curl -s -X POST "http://localhost:8000/data/download?ticker=GBPUSD%3DX&interval=1h&days=730"
```

This fetches 365 days of 15-min bars by splitting into **7 × 59-day chunks** (yfinance enforces a 60-day limit per sub-hourly request), merges and deduplicates them, and writes the result to `data/raw/<ticker>_<interval>.csv`.

---

## Preprocessing Pipeline

The master pipeline (`pipeline.py`) chains five sklearn-compatible transformers:

```
raw DataFrame
    │
    ▼  CleaningBridge        Strip/lowercase columns → sort by time (UTC) →
    │                        deduplicate timestamps → coerce OHLCV to float →
    │                        OHLC sanity check → remove weekends →
    │                        fill 15-min grid gaps (ffill ≤ 2 bars) →
    │                        z-score outlier repair on close price
    │
    ▼  FeatureEngineer       Compute 21 stationary features (see below)
    │
    ▼  WarmupDropper         Drop first 96 rows (rolling-window warm-up period)
    │                        In test/predict mode: drop NaN rows only
    │
    ▼  FeatureSelector       Extract FEATURE_COLS → float32 ndarray or DataFrame
    │
    ▼  FeatureScaler         RobustScaler (fit on train only, transform on val/test)
    │
    ▼  float32 ndarray  (n_samples × 21)
```

**Key design rules:**

- `CleaningBridge` uses `minimal_pipeline()` internally — gap fill and outlier repair run exactly once, not twice
- `WarmupDropper` only truncates the head during **training**; on val/test it only strips NaN rows so no valid prediction rows are lost
- The `RobustScaler` is **fit exclusively on train data** and applied to val/test — no lookahead leakage
- Walk-forward split happens **before** any transformation — raw DataFrames are split, then processed independently

---

## Feature Engineering

21 stationary, look-back-only features — nothing that looks ahead:

| Category | Features |
|---|---|
| **Price action** | `returns`, `log_returns`, `hl_range`, `body_ratio` |
| **Momentum** | `rsi_14`, `macd_hist`, `stoch_k` |
| **Trend** | `ema_ratio` (EMA12/EMA26 − 1) |
| **Volatility** | `atr_ratio`, `bb_pct`, `bb_width` |
| **Rolling vol** | `roll_std_16`, `roll_std_96` |
| **Lag returns** | `return_lag_1`, `return_lag_4`, `return_lag_16` |
| **Time** | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` |
| **Session** | `is_overlap` (London/NY overlap 13–16 UTC) |

**Top 5 by Random Forest importance:** `return_lag_1` · `atr_ratio` · `ema_ratio` · `rsi_14` · `roll_std_96`

### Why these and not more?

- Raw price lags (e.g. `close_lag_1`) are non-stationary — **return lags** carry the same signal without drift
- `ema12` / `ema26` raw values are replaced by **`ema_ratio`** — scale-free and stationary
- `atr_14` raw is replaced by **`atr_ratio`** (ATR / close) — comparable across years
- Three rolling windows rather than six — reduces noise without losing signal
- `is_overlap` only (not `is_london`, `is_ny` separately) — EUR/USD volatility concentrates in the 13–16 UTC window; individual session flags add correlated noise

---

## Model

A **soft-voting ensemble** of three calibrated classifiers:

```python
VotingClassifier(
    estimators = [
        ("rf",  CalibratedClassifierCV(RandomForestClassifier(...))),    # weight 2
        ("hgb", CalibratedClassifierCV(HistGradientBoostingClassifier(...))), # weight 2
        ("lr",  LogisticRegression(...)),                                 # weight 1
    ],
    voting = "soft",
    weights = [2, 2, 1],
)
```

| Component | Role |
|---|---|
| `RandomForest` (300 trees, depth 8) | Non-linear interactions, low variance |
| `HistGradientBoosting` (400 iters, depth 5) | Sequential boosting, strongest on tabular |
| `LogisticRegression` (C=0.05) | Linear regularising anchor, prevents overfit |
| `CalibratedClassifierCV(isotonic, cv=3)` | Makes `predict_proba()` reliable for both RF and HGB |

**Target:** `1` = next candle GREEN (close > open), `0` = RED (close ≤ open)

**Class balance:** ~50% GREEN / ~50% RED — no class weighting issues

---

## Performance

Evaluated on a held-out **chronological test set** (last 20% of data — never seen during training):

| Metric | Validation | Test |
|---|---|---|
| Accuracy | 51.58% | **51.61%** |
| F1 Score | 0.3858 | **0.4855** |
| ROC-AUC | 0.5117 | **0.5219** |

> **Note on expectations:** Forex candle-direction is near-random. Any model consistently above 52% accuracy is statistically significant and profitable after spread on many strategies. The baseline (random guessing) is exactly 50%.

---

## API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### Info Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Welcome message + route map |
| `GET` | `/health` | Liveness, artifact status, pipeline steps, uptime |
| `GET` | `/metrics` | Stored val/test accuracy, F1, ROC-AUC |
| `GET` | `/features` | Feature list + Random Forest importances (ranked) |

### Prediction Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/predict/latest` | **Live prediction** — fetches fresh yfinance data autonomously |
| `POST` | `/predict` | Single prediction from JSON candle array |
| `POST` | `/predict/csv` | Single prediction from uploaded CSV file |
| `POST` | `/predict/batch` | All-row predictions from JSON (backtest) |
| `POST` | `/predict/batch/csv` | All-row predictions from uploaded CSV |

### Data Routes

| Method | Route | Description |
|---|---|---|
| `POST` | `/data/download` | Download 365-day OHLCV from yfinance → save to CSV |

### Response Schemas

**`GET /predict/latest` → `LivePredictionOut`**

```json
{
  "next_candle_colour": "RED",
  "signal":             0,
  "green_prob":         0.486710,
  "red_prob":           0.513290,
  "confidence":         0.513290,
  "current_candle": {
    "time":   "2026-02-26T19:15:00+00:00",
    "open":   1.18032,
    "high":   1.18080,
    "low":    1.18031,
    "close":  1.18046,
    "volume": 1166.0,
    "colour": "RED"
  },
  "current_colour":  "RED",
  "ticker":          "EURUSD=X",
  "interval":        "15m",
  "n_candles_used":  400,
  "fetched_at":      "2026-03-06T08:16:03Z",
  "latency_ms":      312.4
}
```

**`POST /predict` → `PredictionOut`**

```json
{
  "colour":      "RED",
  "green_prob":  0.486710,
  "red_prob":    0.513290,
  "confidence":  0.513290,
  "signal":      0,
  "last_candle": "2026-02-26T19:15:00+05:30",
  "latency_ms":  289.1
}
```

**`POST /data/download` → `DataDownloadOut`**

```json
{
  "ticker":          "EURUSD=X",
  "interval":        "15m",
  "days":            365,
  "n_bars":          25057,
  "date_from":       "2025-03-06 08:30:00+00:00",
  "date_to":         "2026-03-06 19:15:00+00:00",
  "csv_path":        "/project/data/raw/EURUSD_15m.csv",
  "file_size_kb":    2802.8,
  "chunks_fetched":  7,
  "latency_ms":      18412.3
}
```

---

## Sample Requests

### Info

```bash
curl http://localhost:8000/health | python3 -m json.tool
curl http://localhost:8000/metrics | python3 -m json.tool
curl http://localhost:8000/features | python3 -m json.tool
```

### Live prediction (no file needed)

```bash
# Default: EURUSD=X, 15m bars, last 5 days
curl "http://localhost:8000/predict/latest"

# GBPUSD, 1-hour bars
curl "http://localhost:8000/predict/latest?ticker=GBPUSD%3DX&interval=1h&period=60d"
```

### JSON payload

```bash
# Generate candles.json from your CSV
python3 -c "
import pandas as pd, json
df = pd.read_csv('data/raw/OANDA_EURUSD_15.csv')
json.dump({'candles': df.tail(125).to_dict(orient='records')}, open('candles.json','w'))
"

# Single prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @candles.json | python3 -m json.tool

# Batch (all rows)
curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -d @candles.json | python3 -m json.tool
```

### CSV upload

```bash
# Single prediction
curl -X POST http://localhost:8000/predict/csv \
     -F "file=@data/raw/OANDA_EURUSD_15.csv;type=text/csv" | python3 -m json.tool

# Batch backtest
curl -X POST http://localhost:8000/predict/batch/csv \
     -F "file=@data/raw/OANDA_EURUSD_15.csv;type=text/csv" | python3 -m json.tool
```

### Download fresh data

```bash
# 365 days of EUR/USD 15m → data/raw/EURUSD_15m.csv
curl -X POST "http://localhost:8000/data/download" | python3 -m json.tool

# 730 days of GBP/USD 1-hour bars
curl -X POST "http://localhost:8000/data/download?ticker=GBPUSD%3DX&interval=1h&days=730"
```

### Save batch results to CSV

```bash
curl -s -X POST http://localhost:8000/predict/batch/csv \
  -F "file=@data/raw/OANDA_EURUSD_15.csv;type=text/csv" \
  | python3 -c "
import sys, json, csv
body = json.load(sys.stdin)
keys = ['candle_time','signal','colour','green_prob','red_prob','confidence']
w = csv.DictWriter(sys.stdout, fieldnames=keys, extrasaction='ignore')
w.writeheader(); w.writerows(body['predictions'])
" > predictions.csv
```

### Poll every 15 minutes

```bash
watch -n 900 'curl -s http://localhost:8000/predict/latest | python3 -m json.tool'
```

---

## Testing

```bash
# Run full suite (57 tests)
pytest test_api.py -v

# Single class
pytest test_api.py -v -k "health"
pytest test_api.py -v -k "Batch"

# Short output
pytest test_api.py -v --tb=short
```

### Test coverage

| Class | Tests | What's covered |
|---|---|---|
| `TestRoot` | 4 | Welcome page, route map, docs link |
| `TestHealth` | 7 | Status ok, artifacts loaded, pipeline step order, feature count, uptime |
| `TestMetrics` | 5 | Accuracy/F1/AUC ranges, exact match with `metrics.json`, AUC > 0.50 |
| `TestFeatures` | 6 | 21 features, known names present, importances sum to 1 |
| `TestPredictLatest` | 6 | Schema, tail param, tail<120 → 422, determinism |
| `TestPredictJSON` | 8 | Schema, last_candle match, too-few → 422, invalid OHLC → 422, negative price → 422 |
| `TestPredictCSV` | 6 | Happy path, too-few rows → 422, corrupt CSV, missing column |
| `TestPredictBatchJSON` | 8 | Schema, n_predictions, every-row probs sum to 1, latency positive |
| `TestPredictBatchCSV` | 7 | Both colours appear, corrupt CSV error, too-few → 422 |
| `TestUnknownRoutes` | 4 | 404 on unknown paths, 405 on wrong method |
| `TestCrossRouteConsistency` | 4 | JSON == CSV, last batch == single, feature counts agree |

---

## Configuration

All tuneable constants are at the top of each file — no separate config file needed:

| File | Constant | Default | Description |
|---|---|---|---|
| `api.py` | `MIN_CONTEXT_ROWS` | `120` | Minimum candles required per request |
| `api.py` | `_YF_TICKER` | `EURUSD=X` | Default yfinance ticker |
| `api.py` | `_YF_INTERVAL` | `15m` | Default bar interval |
| `api.py` | `_YF_PERIOD` | `5d` | Default lookback for `/predict/latest` |
| `api.py` | `_CHUNK_DAYS` | `59` | Max days per yfinance chunk (hard limit: 60) |
| `pipeline.py` | `walk_forward_split` | `0.70 / 0.10` | Train/val ratio |
| `preprocessing/cleaning.py` | `_FREQ` | `15min` | Expected bar frequency |
| `preprocessing/cleaning.py` | `ffill_limit` | `2` | Max bars to forward-fill across gaps |
| `train.py` | `warmup_bars` | `96` | Rolling-window warm-up bars to drop |

---

## Workflow

### Retrain on new data

```bash
# Option A: download fresh data then retrain in two commands
curl -X POST "http://localhost:8000/data/download"
python run.py train

# Option B: use your own CSV, retrain from scratch
python run.py train --csv data/raw/my_new_data.csv
```

### Change instrument

```bash
# Download GBP/USD, retrain, serve
curl -X POST "http://localhost:8000/data/download?ticker=GBPUSD%3DX&filename=gbpusd_15m"
python run.py train --csv data/raw/gbpusd_15m.csv
python run.py serve
```

### Development mode (hot reload)

```bash
python run.py serve --reload --port 8000
```

### Inference without the server

```python
from train import predict_next_candle
import pandas as pd

df     = pd.read_csv("data/raw/OANDA_EURUSD_15.csv").tail(150)
result = predict_next_candle(df)

print(result["colour"])      # "RED" or "GREEN"
print(result["confidence"])  # e.g. 0.5133
```

---

## run.py Reference

```
python run.py <mode> [options]

Modes
  train     Preprocess raw CSV + train model + save artifacts
  serve     Start FastAPI server (artifacts must exist)
  all       Train then immediately serve
  predict   CLI prediction from the latest candles (no server needed)

Options
  --csv PATH      Raw CSV path          (default: data/raw/OANDA_EURUSD_15.csv)
  --host HOST     API bind address      (default: 0.0.0.0)
  --port PORT     API port              (default: 8000)
  --reload        Enable uvicorn auto-reload
  --tail N        Context rows for CLI predict (default: 300)
```
