from __future__ import annotations

import importlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any
import yfinance as yf

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from src.models.models import MIN_CONTEXT_ROWS

def _df_from_candles(candles: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([c.model_dump() for c in candles])


def _df_from_csv_bytes(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse CSV: {exc}",
        )

def _validate_size(df: pd.DataFrame, min_context_rows: int) -> None:
    if len(df) < min_context_rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Need at least {min_context_rows} candles after cleaning; "
                f"received {len(df)}."
            ),
        )

def _run_pipeline(df: pd.DataFrame, pipe: Any) -> np.ndarray:
    try:
        X = pipe.transform(df)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Preprocessing failed: {exc}",
        )
    if X.shape[0] == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Pipeline produced zero usable rows - send more context candles.",
        )
    return X

def _predict_single(X: np.ndarray, last_time: str, t0: float, model: Any) -> dict:
    row = X[-1:]
    signal = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    green_p = float(proba[1])
    red_p = float(proba[0])
    colour = "GREEN" if signal == 1 else "RED"

    return {
        "colour": colour,
        "green_prob": round(green_p, 6),
        "red_prob": round(red_p, 6),
        "confidence": round(green_p if signal == 1 else red_p, 6),
        "signal": signal,
        "last_candle": last_time,
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }

def _predict_batch(X: np.ndarray, times: list[str], t0: float, model: Any) -> dict:
    signals = model.predict(X).tolist()
    probas = model.predict_proba(X)

    predictions = [
        {
            "candle_time": times[i] if i < len(times) else "",
            "signal": int(signals[i]),
            "colour": "GREEN" if signals[i] == 1 else "RED",
            "green_prob": round(float(probas[i, 1]), 6),
            "red_prob": round(float(probas[i, 0]), 6),
            "confidence": round(float(probas[i, signals[i]]), 6),
        }
        for i in range(len(signals))
    ]

    return {
        "n_predictions": len(predictions),
        "predictions": predictions,
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }

def _extract_times(df: pd.DataFrame, n: int) -> list[str]:
    try:
        col = next(c for c in ("time", "Time", "DATE", "date") if c in df.columns)
        return df[col].astype(str).tail(n).tolist()
    except StopIteration:
        return [""] * n

def _register_pickle_module_aliases() -> None:
    if "preprocessing" not in sys.modules:
        sys.modules["preprocessing"] = importlib.import_module("src.preprocessing")
    if "preprocessing.cleaning" not in sys.modules:
        sys.modules["preprocessing.cleaning"] = importlib.import_module(
            "src.preprocessing.cleaning"
        )
    if "preprocessing.feature_pipe" not in sys.modules:
        sys.modules["preprocessing.feature_pipe"] = importlib.import_module(
            "src.preprocessing.feature_pipe"
        )

def _load_artifacts(
    registry: Any, master_pipe_path: Path, model_path: Path, metrics_path: Path
) -> None:
    _register_pickle_module_aliases()

    for path in (master_pipe_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

    try:
        registry.pipe = joblib.load(master_pipe_path)
        registry.model = joblib.load(model_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Failed to load artifacts due to missing module: {exc}. "
            "Ensure all preprocessing modules are properly installed and importable."
        )

    registry.pipe.set_params(warmup__apply_on_transform=False)

    if metrics_path.exists():
        with open(metrics_path) as f:
            registry.metrics = json.load(f)

    registry.loaded = True
    registry.load_ts = time.time()

def _fetch_yfinance(
    ticker :str,
    interval :str,
    period :str
) -> pd.DataFrame:
    interval_norm = (interval or "").strip().lower()
    period_candidates = [period]

    if interval_norm == "1m":
        period_candidates += ["7d"]
    elif interval_norm in {"2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
        period_candidates += ["1mo", "2mo"]
    else:
        period_candidates += ["1mo", "3mo"]

    tried_periods: list[str] = []
    insufficient_rows: list[str] = []
    last_download_error: str | None = None

    for candidate_period in period_candidates:
        if candidate_period in tried_periods:
            continue
        tried_periods.append(candidate_period)

        raw = None
        for attempt in range(2):
            try:
                raw = yf.download(
                    tickers      = ticker,
                    interval     = interval,
                    period       = candidate_period,
                    auto_adjust  = True,
                    progress     = False,
                    multi_level_index= False,
                )
            except Exception as exc:
                last_download_error = str(exc)
                raw = None

            if raw is not None and not raw.empty:
                break
            if attempt == 0:
                time.sleep(0.8)

        if raw is None or raw.empty:
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw.reset_index()
        idx_col = next(
            (c for c in df.columns if c.lower() in ("datetime", "date")),
            df.columns[0],
        )
        df = df.rename(columns={
            idx_col:  "time",
            "Open":   "open",
            "High":   "high",
            "Low":    "low",
            "Close":  "close",
        })

        if "Volume" not in df.columns:
            df["Volume"] = 0.0
        df["Volume"] = df["Volume"].fillna(0.0)

        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

        if len(df) >= MIN_CONTEXT_ROWS:
            return df

        insufficient_rows.append(f"{candidate_period}:{len(df)}")

    detail = (
        f"yfinance returned no usable data for ticker='{ticker}' interval='{interval}'. "
        f"Tried periods: {', '.join(tried_periods)}."
    )
    if insufficient_rows:
        detail += (
            f" Returned too few rows for context (need {MIN_CONTEXT_ROWS}): "
            f"{'; '.join(insufficient_rows)}."
        )
    else:
        detail += " No rows returned."
    if last_download_error:
        detail += f" Last yfinance error: {last_download_error}"

    raise HTTPException(status_code=503, detail=detail)
