from __future__ import annotations

import numpy as np
import pandas as pd

from ..ta_primitives import ema, normalize, rescale, rma, sma
from .base import Indicator
from .registry import register_indicator


def _n_rsi(close: pd.Series, length: int, smooth: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rescale(ema(rsi, smooth), 0, 100, 0, 1)


def _n_cci(close: pd.Series, length: int, smooth: int) -> pd.Series:
    ma = sma(close, length)
    mad = (close - ma).abs().rolling(length, min_periods=1).mean()
    cci = (close - ma) / (0.015 * mad.replace(0, np.nan))
    return normalize(ema(cci, smooth), 0, 1)


def _n_wt(hlc3: pd.Series, n1: int, n2: int) -> pd.Series:
    esa = ema(hlc3, n1)
    d = ema((hlc3 - esa).abs(), n1)
    ci = (hlc3 - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, 4)
    return normalize(wt1 - wt2, 0, 1)


def _n_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr = (pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1)).max(axis=1)
    plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0.0).clip(lower=0)
    minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0.0).clip(lower=0)
    tr_s = rma(tr, length)
    plus_s = rma(plus_dm, length)
    minus_s = rma(minus_dm, length)
    di_plus = plus_s / tr_s.replace(0, np.nan) * 100
    di_minus = minus_s / tr_s.replace(0, np.nan) * 100
    dx = (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan) * 100
    return rescale(rma(dx, length), 0, 100, 0, 1)


@register_indicator
class PineFeatureIndicator(Indicator):
    name = "pine_feature"

    def compute(self, df: pd.DataFrame, feature: str, param_a: int, param_b: int = 1, source: str = "close") -> pd.DataFrame:
        close = df[source].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        hlc3 = (high + low + close) / 3
        feature_u = feature.upper()
        if feature_u == "RSI":
            values = _n_rsi(close, param_a, param_b)
        elif feature_u == "WT":
            values = _n_wt(hlc3, param_a, param_b)
        elif feature_u == "CCI":
            values = _n_cci(close, param_a, param_b)
        elif feature_u == "ADX":
            values = _n_adx(high, low, close, param_a)
        else:
            raise ValueError(f"Unsupported Pine feature: {feature}")
        return pd.DataFrame({f"{feature_u.lower()}_{param_a}_{param_b}": values})
