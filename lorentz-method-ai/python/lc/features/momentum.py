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
    """Pine-compatible ADX feature using recursive Wilder smoothing.

    Matches MLExtensions.n_adx which uses:
        trSmooth := nz(trSmooth[1]) - nz(trSmooth[1]) / length + tr
    instead of ta.rma (which initializes with SMA).
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    n = len(h)
    tr_smooth = np.zeros(n)
    plus_smooth = np.zeros(n)
    minus_smooth = np.zeros(n)
    dx_arr = np.full(n, np.nan)

    for i in range(n):
        prev_c = c[i - 1] if i > 0 else c[i]
        tr_val = max(h[i] - l[i], abs(h[i] - prev_c), abs(l[i] - prev_c))
        prev_h = h[i - 1] if i > 0 else h[i]
        prev_l = l[i - 1] if i > 0 else l[i]
        up_move = h[i] - prev_h
        down_move = prev_l - l[i]
        plus_dm = max(up_move, 0.0) if up_move > down_move else 0.0
        minus_dm = max(down_move, 0.0) if down_move > up_move else 0.0
        # Pine: trSmooth := nz(trSmooth[1]) - nz(trSmooth[1]) / length + tr
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / length + tr_val if i > 0 else tr_val
        plus_smooth[i] = plus_smooth[i - 1] - plus_smooth[i - 1] / length + plus_dm if i > 0 else plus_dm
        minus_smooth[i] = minus_smooth[i - 1] - minus_smooth[i - 1] / length + minus_dm if i > 0 else minus_dm
        if tr_smooth[i] != 0:
            di_p = plus_smooth[i] / tr_smooth[i] * 100
            di_n = minus_smooth[i] / tr_smooth[i] * 100
            denom = di_p + di_n
            dx_arr[i] = abs(di_p - di_n) / denom * 100 if denom != 0 else 0.0
        else:
            dx_arr[i] = 0.0

    adx = rma(pd.Series(dx_arr), length)
    return rescale(adx, 0, 100, 0, 1)


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
