from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for alias in ("vol", "tick_volume"):
        if alias in df.columns and "volume" not in df.columns:
            df = df.rename(columns={alias: "volume"})
    return df

def _sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" not in df.columns:
        raise ValueError("Expected a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df = df.dropna(subset=["time"])
    return df.sort_values("time").reset_index(drop=True)

def _dedup_time(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=cols).reset_index(drop=True)

def _sanitize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    bad = (
        (h < np.maximum.reduce([o, c, l]))
        | (l > np.minimum.reduce([o, c, h]))
        | (df[["open", "high", "low", "close"]].le(0).any(axis=1))
    )
    return df.loc[~bad].reset_index(drop=True)

def minimal_pipeline(
    *,
    keep_time: bool = True,
    keep_raw_ohlcv: bool = True,
    required_cols: list[str] | None = None,
) -> Pipeline:
    cols = required_cols or ["open", "high", "low", "close", "volume"]

    def _select(df: pd.DataFrame) -> pd.DataFrame:
        out: list[str] = []
        if keep_time:
            out.append("time")
        if keep_raw_ohlcv:
            out += [c for c in cols if c in df.columns]
        if not out:
            raise ValueError("Nothing selected — check keep_time / keep_raw_ohlcv flags.")
        return df[out].copy()

    return Pipeline([
        ("standardize_cols",  FunctionTransformer(_standardize_columns, validate=False)),
        ("sort_time",         FunctionTransformer(_sort_by_time,        validate=False)),
        ("dedup_time",        FunctionTransformer(_dedup_time,          validate=False)),
        ("coerce_numeric",    FunctionTransformer(lambda d: _coerce_numeric(d, cols), validate=False)),
        ("sanitize_ohlc",     FunctionTransformer(_sanitize_ohlc,       validate=False)),
        ("select",            FunctionTransformer(_select,              validate=False)),
    ])

_FREQ = "15min"
_WEEKEND_DAYS = {5, 6}

def _remove_weekends(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["time"].dt.dayofweek.isin(_WEEKEND_DAYS)
    return df.loc[~mask].reset_index(drop=True)

def _fill_gaps(df: pd.DataFrame, freq: str = _FREQ, ffill_limit: int = 2) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("time")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    df = df.reindex(full_idx)
    df.ffill(limit=ffill_limit, inplace=True)
    df.dropna(inplace=True)
    df.index.name = "time"
    return df.reset_index()

def _fix_outliers(df: pd.DataFrame, col: str = "close", z_thresh: float = 4.0) -> pd.DataFrame:
    df = df.copy()
    rets = df[col].pct_change().fillna(0)
    z    = np.abs(stats.zscore(rets))
    bad  = z > z_thresh
    if bad.any():
        roll_med = df[col].rolling(5, center=True, min_periods=1).median()
        df.loc[bad, col] = roll_med[bad]
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"]  = df[["open", "low",  "close"]].min(axis=1)
    return df

def _trim_warmup(df: pd.DataFrame, warmup: int = 96) -> pd.DataFrame:
    return df.iloc[warmup:].dropna().reset_index(drop=True)

def _scale_features(
    df: pd.DataFrame,
    exclude: tuple[str, ...] = ("time", "open", "high", "low", "close", "volume"),
) -> pd.DataFrame:
    df = df.copy()
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_float_dtype(df[c])]
    scaler = RobustScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])
    return df

def enhanced_pipeline(
    *,
    freq: str = _FREQ,
    ffill_limit: int = 2,
    outlier_z: float = 4.0,
    warmup_bars: int = 96,
    scale: bool = True,
    required_cols: list[str] | None = None,
) -> Pipeline:
    cols = required_cols or ["open", "high", "low", "close", "volume"]

    steps = [
        ("standardize_cols", FunctionTransformer(_standardize_columns,      validate=False)),
        ("sort_time",        FunctionTransformer(_sort_by_time,             validate=False)),
        ("dedup_time",       FunctionTransformer(_dedup_time,               validate=False)),
        ("coerce_numeric",   FunctionTransformer(lambda d: _coerce_numeric(d, cols), validate=False)),
        ("sanitize_ohlc",    FunctionTransformer(_sanitize_ohlc,            validate=False)),

        ("remove_weekends",  FunctionTransformer(_remove_weekends,          validate=False)),
        ("fill_gaps",        FunctionTransformer(
            lambda d: _fill_gaps(d, freq=freq, ffill_limit=ffill_limit),   validate=False)),
        ("fix_outliers",     FunctionTransformer(
            lambda d: _fix_outliers(d, z_thresh=outlier_z),                validate=False)),
        ("trim_warmup",      FunctionTransformer(
            lambda d: _trim_warmup(d, warmup=warmup_bars),                 validate=False)),
    ]

    if scale:
        steps.append(
            ("scale_features", FunctionTransformer(_scale_features, validate=False))
        )

    return Pipeline(steps)