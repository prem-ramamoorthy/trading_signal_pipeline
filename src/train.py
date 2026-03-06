from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore")

from pipeline import build_master_pipeline, walk_forward_split
from preprocessing.cleaning import (
    _fill_gaps,
    _fix_outliers,
    _remove_weekends,
    minimal_pipeline,
)
from preprocessing.feature_pipe import (
    FEATURE_COLS,
    FeatureEngineer,
    WarmupDropper,
)

ARTIFACTS_DIR = Path("/tmp/artifacts")
PROCESSED_DIR = Path("/tmp/processed")
RAW_CSV       = Path("data/raw/OANDA_EURUSD_15.csv")

def build_Xy(
    raw_df: pd.DataFrame,
    *,
    is_train: bool = True,
    scaler: RobustScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    cleaner = minimal_pipeline()
    df = cleaner.fit_transform(raw_df)
    df = _remove_weekends(df)
    df = _fill_gaps(df)
    df = _fix_outliers(df)

    df = FeatureEngineer().transform(df)

    df["target"] = (df["close"].shift(-1) > df["open"].shift(-1)).astype("Int8")
    df = df.dropna(subset=["target"])

    df = WarmupDropper(n_bars=96, apply_on_transform=is_train).transform(df)

    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.int8)

    if fit_scaler:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)
    else:
        raise ValueError("Either pass scaler= or set fit_scaler=True.")

    return X, y, scaler


def build_model() -> VotingClassifier:
    rf = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=30,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            ccp_alpha=0.02,
        ),
        cv=3,
        method="isotonic",
    )

    hgb = CalibratedClassifierCV(
        HistGradientBoostingClassifier(
            max_iter=400,
            max_depth=5,
            learning_rate=0.01,
            min_samples_leaf=30,
            l2_regularization=0.6,
            random_state=42,
        ),
        cv=3,
        method="isotonic",
    )

    lr = LogisticRegression(
        C=0.06,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    return VotingClassifier(
        estimators=[("rf", rf), ("hgb", hgb), ("lr", lr)],
        voting="soft",
        weights=[2, 2, 1],
        n_jobs=-1,
    )

def evaluate(
    model: VotingClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> dict:
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    f1  = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_proba)
    cm  = confusion_matrix(y, y_pred)

    print(f"\n{'─'*52}")
    print(f"  {split_name.upper()} EVALUATION")
    print(f"{'─'*52}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              RED    GREEN")
    print(f"  Actual RED  {cm[0,0]:>6}  {cm[0,1]:>6}")
    print(f"  Actual GRN  {cm[1,0]:>6}  {cm[1,1]:>6}")
    print(f"\n{classification_report(y, y_pred, target_names=['RED','GREEN'], zero_division=0)}")

    return {"split": split_name, "accuracy": acc, "f1": f1, "roc_auc": auc,
            "confusion_matrix": cm.tolist()}

def train(csv_path: Path = RAW_CSV) -> dict:
    print("═" * 52)
    print("  EURUSD 15-MIN  ·  NEXT CANDLE COLOUR CLASSIFIER")
    print("═" * 52)

    train_pkl = PROCESSED_DIR / "train_raw.pkl"
    val_pkl   = PROCESSED_DIR / "val_raw.pkl"
    test_pkl  = PROCESSED_DIR / "test_raw.pkl"

    if train_pkl.exists() and val_pkl.exists() and test_pkl.exists():
        print("\n[1/5] Loading existing processed splits …")
        train_raw = joblib.load(train_pkl)
        val_raw   = joblib.load(val_pkl)
        test_raw  = joblib.load(test_pkl)
    else:
        print(f"\n[1/5] Splits not found — splitting raw CSV: {csv_path}")
        raw = pd.read_csv(csv_path)
        train_raw, val_raw, test_raw = walk_forward_split(raw, 0.70, 0.10)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(train_raw, train_pkl)
        joblib.dump(val_raw,   val_pkl)
        joblib.dump(test_raw,  test_pkl)

    print(f"     train={len(train_raw):,}  val={len(val_raw):,}  test={len(test_raw):,} rows")

    print("\n[2/5] Building feature matrices and targets …")
    X_train, y_train, scaler = build_Xy(train_raw, is_train=True,  fit_scaler=True)
    X_val,   y_val,   _      = build_Xy(val_raw,   is_train=False, scaler=scaler)
    X_test,  y_test,  _      = build_Xy(test_raw,  is_train=False, scaler=scaler)

    print(f"     X_train {X_train.shape}  green={y_train.mean():.3f}")
    print(f"     X_val   {X_val.shape}  green={y_val.mean():.3f}")
    print(f"     X_test  {X_test.shape}  green={y_test.mean():.3f}")

    print("\n[3/5] Training ensemble (RF + HGB + LR) …")
    model = build_model()
    model.fit(X_train, y_train)
    print("     Done.")

    print("\n[4/5] Evaluating …")
    val_metrics  = evaluate(model, X_val,  y_val,  "validation")
    test_metrics = evaluate(model, X_test, y_test, "test")

    print("\n[5/5] Saving artefacts …")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path   = ARTIFACTS_DIR / "candle_colour_model.pkl"
    scaler_path  = ARTIFACTS_DIR / "feature_scaler.pkl"
    metrics_path = ARTIFACTS_DIR / "metrics.json"

    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)

    metrics = {
        "features"  : FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "validation": {k: v for k, v in val_metrics.items()  if k != "confusion_matrix"},
        "test"      : {k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"     Model   → {model_path}")
    print(f"     Scaler  → {scaler_path}")
    print(f"     Metrics → {metrics_path}")

    print(f"\n{'═'*52}")
    print(f"  FINAL TEST  accuracy={test_metrics['accuracy']:.4f}  "
          f"f1={test_metrics['f1']:.4f}  auc={test_metrics['roc_auc']:.4f}")
    print(f"{'═'*52}\n")

    return {
        "model"   : model,
        "scaler"  : scaler,
        "metrics" : metrics,
        "X_train" : X_train,
        "X_val"   : X_val,
        "X_test"  : X_test,
        "y_train" : y_train,
        "y_val"   : y_val,
        "y_test"  : y_test,
    }

def predict_next_candle(
    context_df: pd.DataFrame,
    *,
    model_path: Path = ARTIFACTS_DIR / "candle_colour_model.pkl",
    scaler_path: Path = ARTIFACTS_DIR / "feature_scaler.pkl",
) -> dict:
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    cleaner = minimal_pipeline()
    df = cleaner.fit_transform(context_df)
    df = _remove_weekends(df)
    df = _fill_gaps(df)
    df = _fix_outliers(df)
    df = FeatureEngineer().transform(df)
    df = WarmupDropper(n_bars=96, apply_on_transform=False).transform(df)

    if df.empty:
        raise ValueError(
            "Not enough context rows after cleaning — pass at least 120 raw candles."
        )

    X_live = df[FEATURE_COLS].iloc[[-1]].to_numpy(dtype=np.float32)
    X_live = scaler.transform(X_live)

    pred      = int(model.predict(X_live)[0])
    proba     = model.predict_proba(X_live)[0]
    green_p   = float(proba[1])
    red_p     = float(proba[0])

    return {
        "colour"     : "GREEN" if pred == 1 else "RED",
        "probability": green_p if pred == 1 else red_p,
        "green_prob" : green_p,
        "red_prob"   : red_p,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train next-candle colour classifier.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RAW_CSV,
        help="Path to raw OANDA EUR/USD 15-min CSV (default: data/raw/OANDA_EURUSD_15.csv)",
    )
    args = parser.parse_args()
    train(csv_path=args.csv)
