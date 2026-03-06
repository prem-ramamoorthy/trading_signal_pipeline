from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

FEATURE_COLS: list[str] = [
    "returns", "log_returns", "hl_range", "body_ratio",
    "rsi_14", "macd_hist", "stoch_k",
    "ema_ratio",
    "atr_ratio", "bb_pct", "bb_width",
    "roll_std_16", "roll_std_96",
    "return_lag_1", "return_lag_4", "return_lag_16",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_overlap",
]

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        c, h, l, o = df["close"], df["high"], df["low"], df["open"]

        df["returns"]    = c.pct_change()
        df["log_returns"]= np.log(c / c.shift(1))
        df["hl_range"]   = h - l
        df["body_ratio"] = (c - o) / df["hl_range"].replace(0, np.nan)

        delta        = c.diff()
        gain         = delta.clip(lower=0).rolling(14).mean()
        loss         = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        ema12           = c.ewm(span=12, adjust=False).mean()
        ema26           = c.ewm(span=26, adjust=False).mean()
        macd            = ema12 - ema26
        df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

        lo14          = l.rolling(14).min()
        hi14          = h.rolling(14).max()
        df["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14).replace(0, np.nan)

        df["ema_ratio"] = (ema12 / ema26.replace(0, np.nan)) - 1

        tr = np.maximum(
            h - l,
            np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()),
        )
        df["atr_ratio"] = tr.rolling(14).mean() / c.replace(0, np.nan)

        bb_mid         = c.rolling(20).mean()
        bb_band        = (4 * c.rolling(20).std()).replace(0, np.nan)
        df["bb_pct"]   = (c - (bb_mid - bb_band / 2)) / bb_band
        df["bb_width"] = bb_band / bb_mid.replace(0, np.nan)

        df["roll_std_16"] = df["returns"].rolling(16).std()
        df["roll_std_96"] = df["returns"].rolling(96).std()

        for lag in (1, 4, 16):
            df[f"return_lag_{lag}"] = df["returns"].shift(lag)

        hr             = df["time"].dt.hour
        dw             = df["time"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * hr / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hr / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * dw / 5)
        df["dow_cos"]  = np.cos(2 * np.pi * dw / 5)

        df["is_overlap"] = hr.between(13, 16).astype(np.int8)

        return df

class WarmupDropper(BaseEstimator, TransformerMixin):

    def __init__(self, n_bars: int = 96, apply_on_transform: bool = True):
        self.n_bars            = n_bars
        self.apply_on_transform= apply_on_transform

    def fit(self, X: pd.DataFrame, y=None) -> "WarmupDropper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.apply_on_transform:
            return X.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        return X.iloc[self.n_bars:].dropna(subset=FEATURE_COLS).reset_index(drop=True)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, cols: list[str] = FEATURE_COLS, return_df: bool = False):
        self.cols      = cols
        self.return_df = return_df

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        missing = [c for c in self.cols if c not in X.columns]
        if missing:
            raise ValueError(f"Features missing after engineering: {missing}")
        return self

    def transform(self, X: pd.DataFrame):
        out = X[self.cols]
        if self.return_df:
            return out
        return out.to_numpy(dtype=np.float32)

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler_ = RobustScaler()

    def fit(self, X, y=None) -> "FeatureScaler":
        self.scaler_.fit(X)
        return self

    def transform(self, X):
        return self.scaler_.transform(X)

def feature_pipeline(
    *,
    warmup_bars: int = 96,
    scale: bool = True,
    return_df: bool = False,
    is_train: bool = True,
) -> Pipeline:
    steps = [
        ("engineer",  FeatureEngineer()),
        ("warmup",    WarmupDropper(n_bars=warmup_bars, apply_on_transform=is_train)),
        ("selector",  FeatureSelector(cols=FEATURE_COLS, return_df=return_df)),
    ]
    if scale:
        steps.append(("scaler", FeatureScaler()))

    return Pipeline(steps)

if __name__ == "__main__":
    pipe    = feature_pipeline(is_train=True, scale=True)
    X_train = pipe.fit_transform(train_df)

    X_val   = pipe.transform(val_df)
    X_test  = pipe.transform(test_df)

    X_live  = pipe.transform(live_context_df)

    print(X_train.shape, X_test.shape)