from __future__ import annotations

import sys
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import datetime as dt
import yfinance as yf

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.pipeline import CleaningBridge
from src.preprocessing.cleaning import (
    _fill_gaps,
    _fix_outliers,
    _remove_weekends,
    minimal_pipeline,
)
from src.preprocessing.feature_pipe import (
    FEATURE_COLS,
    FeatureEngineer,
    WarmupDropper,
)
from src.utils.utils import (
    _df_from_candles,
    _df_from_csv_bytes,
    _extract_times,
    _load_artifacts,
    _predict_batch,
    _predict_single,
    _run_pipeline,
    _validate_size,
    _fetch_yfinance,
)
from src.models.models import (
    MIN_CONTEXT_ROWS,
    BatchPredictionOut,
    Candle,
    CandleList,
    PredictionOut,
    LivePredictionOut,
    DataDownloadOut,
)
from src.pipeline import run
from src.train import argparse, train

BASE_DIR       = Path(__file__).parent
TMP_DIR        = Path("/tmp")
ARTIFACTS_DIR  = TMP_DIR / "artifacts"
PROCESSED_DIR  = TMP_DIR / "processed"
RAW_CSV        = BASE_DIR / "data" / "raw" / "OANDA_EURUSD_15.csv"

MASTER_PIPE_PATH  = ARTIFACTS_DIR / "master_pipe.pkl"
MODEL_PATH        = ARTIFACTS_DIR / "candle_colour_model.pkl"
METRICS_PATH      = ARTIFACTS_DIR / "metrics.json"

_YF_TICKER   = "EURUSD=X"
_YF_INTERVAL = "15m"
_YF_PERIOD   = "5d"

class _Registry:
    pipe:    Any = None
    model:   Any = None
    metrics: dict = {}
    loaded:  bool = False
    load_ts: float = 0.0

registry = _Registry()

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_artifacts(registry, MASTER_PIPE_PATH, MODEL_PATH, METRICS_PATH)
        print(f"[startup] Artifacts loaded from {ARTIFACTS_DIR}")
    except (ModuleNotFoundError, FileNotFoundError) as e1:
        print(f"[startup] Warning: {e1}. Modules may be missing. Some routes may not work.")
        print(f"[startup] Attempting to run training to generate missing artifacts...")
        try:
            run(
                csv_path      = RAW_CSV,
                artifacts_dir = ARTIFACTS_DIR,
                processed_dir = PROCESSED_DIR,
                train_ratio   = 0.70,
                val_ratio     = 0.10,
                freq          = "15min",
                ffill_limit   = 2,
                outlier_z     = 4.0,
                warmup_bars   = 96,
                scale         = True,
            )
            train(csv_path=RAW_CSV)
            _load_artifacts(registry, MASTER_PIPE_PATH, MODEL_PATH, METRICS_PATH)
        except Exception as e2:
            print(f"[startup] Warning: {e2}. Some features may not be available.")
    yield
    print("[shutdown] API shutting down.")

app = FastAPI(
    title       = "EUR/USD Candle Colour Predictor",
    description = (
        "Predicts whether the **next 15-min candle** will be "
        "🟢 GREEN (close > open) or 🔴 RED (close ≤ open). "
        "All routes accept raw OHLCV data and run the full "
        "preprocessing pipeline autonomously."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

@app.get("/", tags=["Info"])
def root():
    return {
        "name"   : "EUR/USD Candle Colour Predictor",
        "version": "1.0.0",
        "docs"   : "/docs",
        "routes" : {
            "GET  /health"              : "Liveness + artifact status",
            "POST /predict"             : "Single prediction from JSON candles",
            "POST /predict/csv"         : "Single prediction from CSV upload",
            "POST /predict/batch"       : "Batch predictions from JSON candles",
            "POST /predict/batch/csv"   : "Batch predictions from CSV upload",
            "GET  /predict/latest"      : "Live prediction via yfinance (no file needed)",
            "GET  /features"            : "Feature list + RF importances",
            "GET  /metrics"             : "Stored val/test metrics",
            "GET  /refresh"             : "Delete old artifacts, rerun pipeline + training, and reload new artifacts",            
        },
    }

@app.get("/health", tags=["Info"])
def health():
    return {
        "status"         : "ok" if registry.loaded else "degraded",
        "artifacts_loaded": registry.loaded,
        "artifacts": {
            "master_pipe"        : MASTER_PIPE_PATH.exists(),
            "candle_colour_model": MODEL_PATH.exists(),
            "metrics"            : METRICS_PATH.exists(),
        },
        "model_steps"     : [s[0] for s in registry.pipe.steps] if registry.loaded else [],
        "n_features"      : len(FEATURE_COLS),
        "uptime_seconds"  : round(time.time() - registry.load_ts, 1) if registry.loaded else 0,
    }

@app.get("/metrics", tags=["Info"])
def get_metrics():
    if not registry.metrics:
        raise HTTPException(status_code=404, detail="metrics.json not found.")
    return registry.metrics

@app.get("/features", tags=["Info"])
def get_features():
    importances = {}
    try:
        rf_cal    = registry.model.estimators_[0]
        rf_base   = rf_cal.calibrated_classifiers_[0].estimator
        imp       = rf_base.feature_importances_
        importances = dict(sorted(
            zip(FEATURE_COLS, imp.tolist()),
            key=lambda x: x[1], reverse=True,
        ))
    except Exception:
        pass

    return {
        "n_features"  : len(FEATURE_COLS),
        "features"    : FEATURE_COLS,
        "importances" : importances,
    }

@app.post(
    "/predict",
    response_model=PredictionOut,
    tags=["Prediction"],
    summary="Single prediction from JSON candles",
    description=(
        f"Send at least **{MIN_CONTEXT_ROWS} raw OHLCV candles** (newest last). "
        "The API runs the full preprocessing pipeline and predicts the **next candle's colour**."
    ),
)
def predict(body: CandleList) -> PredictionOut:
    t0 = time.time()
    df = _df_from_candles(body.candles)
    last_time = str(body.candles[-1].time)
    X  = _run_pipeline(df, registry.pipe)
    return _predict_single(X, last_time, t0, registry.model)

@app.post(
    "/predict/csv",
    response_model=PredictionOut,
    tags=["Prediction"],
    summary="Single prediction from CSV upload",
    description=(
        f"Upload a raw OHLCV CSV file with at least **{MIN_CONTEXT_ROWS} rows**. "
        "Expected columns: `time, open, high, low, close, Volume`. "
        "Returns prediction for the candle **after** the last row."
    ),
)
async def predict_csv(file: UploadFile = File(...)) -> PredictionOut:
    t0   = time.time()
    data = await file.read()
    df   = _df_from_csv_bytes(data)
    _validate_size(df, MIN_CONTEXT_ROWS)
    last_time = str(df.iloc[-1].get("time", ""))
    X = _run_pipeline(df, registry.pipe)
    return _predict_single(X, last_time, t0, registry.model)

@app.post(
    "/predict/batch",
    response_model=BatchPredictionOut,
    tags=["Prediction"],
    summary="Batch predictions from JSON candles",
    description=(
        "Returns a prediction for **every usable row** after the pipeline "
        f"warm-up period (first {MIN_CONTEXT_ROWS} rows consumed). "
        "Useful for backtesting or replaying a candle series."
    ),
)
def predict_batch(body: CandleList) -> BatchPredictionOut:
    t0    = time.time()
    df    = _df_from_candles(body.candles)
    times = [c.time for c in body.candles]
    X     = _run_pipeline(df, registry.pipe)
    aligned_times = times[-X.shape[0]:] if len(times) >= X.shape[0] else [""] * X.shape[0]
    return _predict_batch(X, aligned_times, t0, registry.model)

@app.post(
    "/predict/batch/csv",
    response_model=BatchPredictionOut,
    tags=["Prediction"],
    summary="Batch predictions from CSV upload",
    description=(
        "Upload a CSV and get a prediction for every usable row. "
        "Ideal for offline backtesting — attach your own labels to check accuracy."
    ),
)
async def predict_batch_csv(file: UploadFile = File(...)) -> BatchPredictionOut:
    t0   = time.time()
    data = await file.read()
    df   = _df_from_csv_bytes(data)
    _validate_size(df, MIN_CONTEXT_ROWS)
    times = _extract_times(df, len(df))
    X     = _run_pipeline(df, registry.pipe)
    aligned_times = times[-X.shape[0]:] if len(times) >= X.shape[0] else [""] * X.shape[0]
    return _predict_batch(X, aligned_times, t0, registry.model)

@app.get(
    "/predict/latest",
    response_model=LivePredictionOut,
    tags=["Prediction"],
    summary="Live prediction via yfinance (no file needed)",
    description=(
        "Fetches the latest EUR/USD 15-min candles **live from Yahoo Finance**, "
        "runs the full preprocessing pipeline, and predicts the colour of the "
        "**next candle** relative to the current (last completed) candle.  "
        "No file upload or local CSV required — just call the endpoint.  \n\n"
        "**Query params**  \n"
        f"- `ticker`   Yahoo Finance symbol (default `{_YF_TICKER}`)  \n"
        f"- `interval` Bar size             (default `{_YF_INTERVAL}`)  \n"
        f"- `period`   Lookback window      (default `{_YF_PERIOD}`)  \n"
    ),
)
def predict_latest(
    ticker:   str = Query(default=_YF_TICKER,   description="Yahoo Finance ticker, e.g. EURUSD=X"),
    interval: str = Query(default=_YF_INTERVAL, description="Bar size: 1m 5m 15m 30m 1h"),
    period:   str = Query(default=_YF_PERIOD,   description="Lookback: 1d 5d 1mo — must give ≥120 bars"),
) -> LivePredictionOut:
    t0         = time.time()
    fetched_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    df = _fetch_yfinance(ticker=ticker, interval=interval, period=period)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Asia/Kolkata")
    df["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")
    
    main_data = pd.read_csv("data/raw/OANDA_EURUSD_15.csv")
    main_data = pd.concat([main_data, df], ignore_index=True)
    main_data.drop_duplicates(subset=["time"], inplace=True)
    main_data.to_csv("data/raw/OANDA_EURUSD_15.csv", index=False)
    X = _run_pipeline(df, registry.pipe)

    row    = X[-1:]
    signal = int(registry.model.predict(row)[0])
    proba  = registry.model.predict_proba(row)[0]
    gp     = round(float(proba[1]), 6)
    rp     = round(float(proba[0]), 6)

    last = df.iloc[-1]
    current_colour = "GREEN" if float(last["close"]) > float(last["open"]) else "RED"
    current_candle = {
        "time":   str(last["time"]),
        "open":   round(float(last["open"]),  6),
        "high":   round(float(last["high"]),  6),
        "low":    round(float(last["low"]),   6),
        "close":  round(float(last["close"]), 6),
        "volume": round(float(last["Volume"]), 2),
        "colour": current_colour,
    }

    return LivePredictionOut(
        next_candle_colour = "GREEN" if signal == 1 else "RED",
        signal             = signal,
        green_prob         = gp,
        red_prob           = rp,
        confidence         = round(gp if signal == 1 else rp, 6),
        current_candle     = current_candle,
        current_colour     = current_colour,
        ticker             = ticker,
        interval           = interval,
        n_candles_used     = len(df),
        fetched_at         = fetched_at,
        latency_ms         = round((time.time() - t0) * 1000, 2),
    )

@app.get("/refresh", tags=["Info"])
def refresh_artifacts():
    print("[refresh] Refresh requested. Deleting old artifacts and processed files...")

    try:
        artifacts_deleted = []
        processed_deleted = []

        artifacts_dir = ARTIFACTS_DIR
        processed_dir = PROCESSED_DIR

        artifacts_dir.mkdir(parents=True, exist_ok=True)
        processed_dir.mkdir(parents=True, exist_ok=True)

        for file in artifacts_dir.iterdir():
            if file.is_file():
                file.unlink()
                artifacts_deleted.append(file.name)

        for file in processed_dir.iterdir():
            if file.is_file():
                file.unlink()
                processed_deleted.append(file.name)

        print("[refresh] Old files deleted successfully.")
        print("[refresh] Attempting to regenerate artifacts...")

        registry.pipe = None
        registry.model = None
        registry.metrics = {}
        registry.loaded = False
        registry.load_ts = 0.0

        run(
            csv_path=RAW_CSV,
            artifacts_dir=ARTIFACTS_DIR,
            processed_dir=PROCESSED_DIR,
            train_ratio=0.70,
            val_ratio=0.10,
            freq="15min",
            ffill_limit=2,
            outlier_z=4.0,
            warmup_bars=96,
            scale=True,
        )

        train(csv_path=RAW_CSV)

        _load_artifacts(registry, MASTER_PIPE_PATH, MODEL_PATH, METRICS_PATH)

        print(f"[refresh] Artifacts reloaded successfully from {ARTIFACTS_DIR}")

        return {
            "status": "success",
            "message": "Artifacts refreshed, retrained, and reloaded successfully.",
            "deleted_artifacts": artifacts_deleted,
            "deleted_processed_files": processed_deleted,
            "artifacts_loaded": registry.loaded,
            "model_steps": [s[0] for s in registry.pipe.steps] if registry.loaded else [],
            "n_features": len(FEATURE_COLS),
        }

    except Exception as e:
        print(f"[refresh] Failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Refresh failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API:app", host="0.0.0.0", port=8000, reload=True)
