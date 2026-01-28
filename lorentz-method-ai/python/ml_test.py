"""Feature engineering and regression-based feature ranking.

This script loads an OHLCV CSV, engineers multiple candidate features, and fits
an OLS regression to explain forward returns. It reports the features that
contribute the most (by absolute standardized coefficient) along with
correlations and model fit metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class RegressionSummary:
    coefficients: pd.Series
    correlations: pd.Series
    r2: float
    rows: int


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip().lower() for col in df.columns]
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=2 / (length + 1), adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=1).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return rma(tr, length)


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    features: Dict[str, pd.Series] = {}
    features["return_1"] = close.pct_change()
    features["return_5"] = close.pct_change(5)
    features["log_return"] = np.log(close).diff()
    features["momentum_5"] = close / close.shift(5) - 1
    features["momentum_20"] = close / close.shift(20) - 1
    features["range_pct"] = (high - low) / close

    features["sma_5"] = sma(close, 5) / close - 1
    features["sma_20"] = sma(close, 20) / close - 1
    features["ema_10"] = ema(close, 10) / close - 1
    features["ema_50"] = ema(close, 50) / close - 1
    features["ema_cross"] = ema(close, 10) - ema(close, 50)

    features["volatility_5"] = close.pct_change().rolling(5).std()
    features["volatility_20"] = close.pct_change().rolling(20).std()
    features["atr_14"] = atr(high, low, close, 14)
    features["rsi_14"] = rsi(close, 14)

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    features["volume_zscore"] = (volume - volume_mean) / volume_std.replace(0, np.nan)

    return pd.DataFrame(features)


def prepare_regression_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    features = engineer_features(df)
    target = df["close"].shift(-horizon) / df["close"] - 1
    frame = features.copy()
    frame["target"] = target
    frame = frame.dropna().reset_index(drop=True)
    return frame


def fit_regression(frame: pd.DataFrame) -> RegressionSummary:
    feature_cols = [col for col in frame.columns if col != "target"]
    x = frame[feature_cols]
    y = frame["target"]

    x_std = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
    x_std = x_std.fillna(0.0)

    x_matrix = np.column_stack([np.ones(len(x_std)), x_std.to_numpy()])
    coeffs, _, _, _ = np.linalg.lstsq(x_matrix, y.to_numpy(), rcond=None)

    y_pred = x_matrix @ coeffs
    ss_res = np.sum((y.to_numpy() - y_pred) ** 2)
    ss_tot = np.sum((y.to_numpy() - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    coef_series = pd.Series(coeffs[1:], index=feature_cols).sort_values(key=np.abs, ascending=False)
    corr_series = x.corrwith(y).sort_values(key=np.abs, ascending=False)

    return RegressionSummary(
        coefficients=coef_series,
        correlations=corr_series,
        r2=float(r2),
        rows=len(frame),
    )


def print_summary(summary: RegressionSummary, top_n: int) -> None:
    print("Regression summary")
    print("===================")
    print(f"Rows used: {summary.rows}")
    print(f"R^2: {summary.r2:.4f}")
    print("\nTop coefficients (standardized):")
    print(summary.coefficients.head(top_n).to_string())
    print("\nTop correlations: ")
    print(summary.correlations.head(top_n).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="MYX_FCPO1!, 5.csv",
        help="Path to OHLCV CSV file",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forward return horizon (bars).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="How many top features to display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_ohlcv(args.csv)
    frame = prepare_regression_frame(df, horizon=args.horizon)
    summary = fit_regression(frame)
    print_summary(summary, top_n=args.top)


if __name__ == "__main__":
    main()
