from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.preprocessing.cleaning import (
    minimal_pipeline,
    _remove_weekends,
    _fill_gaps,
    _fix_outliers,
)
from src.preprocessing.feature_pipe import (
    FEATURE_COLS,
    FeatureEngineer,
    FeatureScaler,
    FeatureSelector,
    WarmupDropper,
)

class CleaningBridge(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        freq: str = "15min",
        ffill_limit: int = 2,
        outlier_z: float = 4.0,
    ):
        self.freq        = freq
        self.ffill_limit = ffill_limit
        self.outlier_z   = outlier_z

    def fit(self, X: pd.DataFrame, y=None) -> "CleaningBridge":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cleaner = minimal_pipeline()
        df = cleaner.fit_transform(X)
        df = _remove_weekends(df)
        df = _fill_gaps(df, freq=self.freq, ffill_limit=self.ffill_limit)
        df = _fix_outliers(df, z_thresh=self.outlier_z)
        return df

def build_master_pipeline(
    *,
    freq: str        = "15min",
    ffill_limit: int = 2,
    outlier_z: float = 4.0,
    warmup_bars: int = 96,
    scale: bool      = True,
    return_df: bool  = False,
    is_train: bool   = True,
) -> Pipeline:

    steps: list[tuple[str, BaseEstimator]] = [
        ("cleaning", CleaningBridge(
            freq=freq, ffill_limit=ffill_limit, outlier_z=outlier_z,
        )),
        ("engineer", FeatureEngineer()),
        ("warmup",   WarmupDropper(n_bars=warmup_bars, apply_on_transform=is_train)),
        ("selector", FeatureSelector(cols=FEATURE_COLS, return_df=return_df)),
    ]
    if scale:
        steps.append(("scaler", FeatureScaler()))

    return Pipeline(steps)

def walk_forward_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n     = len(df)
    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:t_end].copy(), df.iloc[t_end:v_end].copy(), df.iloc[v_end:].copy()

def save_object(obj: object, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)
    print(f"  Saved → {p}")

def load_pipeline(path: str | Path) -> Pipeline:
    pipe = joblib.load(Path(path))
    print(f"Pipeline loaded ← {path}")
    return pipe

def run(
    csv_path: str | Path,
    *,
    artifacts_dir: str | Path = "/tmp/artifacts",
    processed_dir: str | Path = "/tmp/processed",
    train_ratio: float = 0.70,
    val_ratio: float   = 0.10,
    **pipeline_kwargs,
) -> dict:
    artifacts_dir = Path(artifacts_dir)
    processed_dir = Path(processed_dir)

    raw = pd.read_csv(csv_path)
    print(f"Loaded   {len(raw):,} raw rows  ←  {csv_path}")

    train_raw, val_raw, test_raw = walk_forward_split(raw, train_ratio, val_ratio)
    print(f"Split    train={len(train_raw):,}  val={len(val_raw):,}  test={len(test_raw):,}")

    pipe    = build_master_pipeline(is_train=True, **pipeline_kwargs)
    X_train = pipe.fit_transform(train_raw)
    print(f"X_train  {X_train.shape}")

    pipe.set_params(warmup__apply_on_transform=False)
    X_val  = pipe.transform(val_raw)
    X_test = pipe.transform(test_raw)
    print(f"X_val    {X_val.shape}")
    print(f"X_test   {X_test.shape}")

    save_object(pipe,      artifacts_dir / "master_pipe.pkl")
    save_object(train_raw, processed_dir / "train_raw.pkl")
    save_object(val_raw,   processed_dir / "val_raw.pkl")
    save_object(test_raw,  processed_dir / "test_raw.pkl")

    return {
        "pipeline" : pipe,
        "X_train"  : X_train,
        "X_val"    : X_val,
        "X_test"   : X_test,
        "features" : FEATURE_COLS,
    }

if __name__ == "__main__":
    result = run(
        csv_path      = Path("data/raw/OANDA_EURUSD_15.csv"),
        artifacts_dir = Path("/tmp/artifacts"),
        processed_dir = Path("/tmp/processed"),
        train_ratio   = 0.70,
        val_ratio     = 0.10,
        freq          = "15min",
        ffill_limit   = 2,
        outlier_z     = 4.0,
        warmup_bars   = 96,
        scale         = True,
    )

    print("\n── Summary ──────────────────────────────────────────")
    print(f"Features   ({len(result['features'])}): {result['features']}")
    print(f"X_train  : {result['X_train'].shape}")
    print(f"X_val    : {result['X_val'].shape}")
    print(f"X_test   : {result['X_test'].shape}")
