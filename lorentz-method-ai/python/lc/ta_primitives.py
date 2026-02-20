from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=2 / (length + 1), adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=1).mean()


def rma(series: pd.Series, length: int) -> pd.Series:
    # Pine Script ta.rma behaves like EMA but alpha = 1/length instead of 2/(length+1),
    # AND it initializes with an SMA of the given length.
    alpha = 1 / length
    sma = series.rolling(window=length, min_periods=length).mean()
    out = np.full(len(series), np.nan)
    for i in range(len(series)):
        if not pd.isna(sma.iloc[i]) and pd.isna(out[i-1]):
            out[i] = sma.iloc[i]
        elif i > 0 and not pd.isna(out[i-1]):
            out[i] = alpha * series.iloc[i] + (1 - alpha) * out[i-1]
    return pd.Series(out, index=series.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return rma(tr, length)


def normalize(src: pd.Series, new_min: float, new_max: float) -> pd.Series:
    hist_min = src.cummin()
    hist_max = src.cummax()
    denom = (hist_max - hist_min).replace(0, 1e-10)
    return new_min + (new_max - new_min) * (src - hist_min) / denom


def rescale(src: pd.Series, old_min: float, old_max: float, new_min: float, new_max: float) -> pd.Series:
    return new_min + (new_max - new_min) * (src - old_min) / max(old_max - old_min, 1e-10)


def barssince(condition: pd.Series) -> pd.Series:
    condition = condition.fillna(False)
    out = np.full(len(condition), np.nan)
    last_true = None
    for i, v in enumerate(condition):
        if bool(v):
            last_true = i
            out[i] = 0
        elif last_true is not None:
            out[i] = i - last_true
    return pd.Series(out, index=condition.index)
