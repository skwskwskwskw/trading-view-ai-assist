from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..features.kernel import gaussian, rational_quadratic
from ..ta_primitives import barssince, ema, rma, sma


@dataclass
class ANNConfig:
    neighbors_count: int = 8
    max_bars_back: int = 2000
    source: str = "close"
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_adx_filter: bool = False
    regime_threshold: float = -0.1
    adx_threshold: int = 20
    use_ema_filter: bool = False
    ema_period: int = 200
    use_sma_filter: bool = False
    sma_period: int = 200
    use_kernel_filter: bool = True
    use_kernel_smoothing: bool = False
    use_dynamic_exits: bool = False
    kernel_lookback: int = 8
    kernel_relative_weight: float = 8.0
    kernel_regression_level: int = 25
    kernel_lag: int = 2


def lorentzian_distance(current: np.ndarray, historical: np.ndarray) -> float:
    return float(np.log1p(np.abs(current - historical)).sum())


def _adx_filter(df: pd.DataFrame, threshold: int, use_filter: bool) -> pd.Series:
    """Pine-compatible ADX filter using recursive Wilder smoothing."""
    if not use_filter:
        return pd.Series(True, index=df.index)
    length = 14
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)
    tr_smooth = np.zeros(n)
    plus_smooth = np.zeros(n)
    minus_smooth = np.zeros(n)
    dx_arr = np.full(n, np.nan)
    adx_arr = np.full(n, np.nan)
    for i in range(n):
        prev_close = close[i - 1] if i > 0 else close[i]
        tr_val = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
        prev_high = high[i - 1] if i > 0 else high[i]
        prev_low = low[i - 1] if i > 0 else low[i]
        up_move = high[i] - prev_high
        down_move = prev_low - low[i]
        plus_dm = max(up_move, 0.0) if up_move > down_move else 0.0
        minus_dm = max(down_move, 0.0) if down_move > up_move else 0.0
        # Pine: trSmooth := nz(trSmooth[1]) - nz(trSmooth[1]) / length + tr
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / length + tr_val if i > 0 else tr_val
        plus_smooth[i] = plus_smooth[i - 1] - plus_smooth[i - 1] / length + plus_dm if i > 0 else plus_dm
        minus_smooth[i] = minus_smooth[i - 1] - minus_smooth[i - 1] / length + minus_dm if i > 0 else minus_dm
        if tr_smooth[i] != 0:
            di_plus = plus_smooth[i] / tr_smooth[i] * 100
            di_minus = minus_smooth[i] / tr_smooth[i] * 100
            denom = di_plus + di_minus
            dx_arr[i] = abs(di_plus - di_minus) / denom * 100 if denom != 0 else 0.0
        else:
            dx_arr[i] = 0.0
    adx_s = pd.Series(dx_arr)
    adx_result = rma(adx_s, length)
    return pd.Series(adx_result.values > threshold, index=df.index)


def _volatility_filter(df: pd.DataFrame, use_filter: bool) -> pd.Series:
    if not use_filter:
        return pd.Series(True, index=df.index)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return rma(tr, 1) > rma(tr, 10)


def _regime_filter(df: pd.DataFrame, threshold: float, use_filter: bool) -> pd.Series:
    if not use_filter:
        return pd.Series(True, index=df.index)
    src = df[["open", "high", "low", "close"]].mean(axis=1)
    value1 = pd.Series(0.0, index=df.index)
    value2 = pd.Series(0.0, index=df.index)
    klmf = pd.Series(0.0, index=df.index)
    for i in range(len(df)):
        if i == 0:
            value1.iloc[i] = 0.2 * (src.iloc[i] - src.iloc[i])
            value2.iloc[i] = 0.1 * (df["high"].iloc[i] - df["low"].iloc[i])
            klmf.iloc[i] = src.iloc[i]
        else:
            value1.iloc[i] = 0.2 * (src.iloc[i] - src.iloc[i - 1]) + 0.8 * value1.iloc[i - 1]
            value2.iloc[i] = 0.1 * (df["high"].iloc[i] - df["low"].iloc[i]) + 0.8 * value2.iloc[i - 1]
            omega = abs(value1.iloc[i] / value2.iloc[i]) if value2.iloc[i] != 0 else 0.0
            alpha = (-omega**2 + np.sqrt(omega**4 + 16 * omega**2)) / 8
            klmf.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * klmf.iloc[i - 1]
    abs_slope = (klmf - klmf.shift(1)).abs()
    baseline = ema(abs_slope, 200)
    decline = (abs_slope - baseline) / baseline.replace(0, np.nan)
    return decline >= threshold


def run_ann(df: pd.DataFrame, feature_columns: Sequence[str], cfg: ANNConfig) -> pd.DataFrame:
    out = df.copy()
    src = out[cfg.source].astype(float)
    y_train = np.where(src.shift(4) < src, -1, np.where(src.shift(4) > src, 1, 0))
    feat_mat = out[list(feature_columns)].to_numpy(dtype=float)

    predictions = np.zeros(len(out), dtype=float)
    signal = np.zeros(len(out), dtype=int)
    max_bars_back_index = max(len(out) - cfg.max_bars_back, 0)

    for bar in range(len(out)):
        # Pine: distances/predictions are non-var => reset every bar
        neighbor_predictions: list[int] = []
        neighbor_distances: list[float] = []
        if bar > 0:
            signal[bar] = signal[bar - 1]
        if bar < max_bars_back_index:
            continue
        last_distance = -1.0
        size_loop = min(cfg.max_bars_back - 1, bar)
        for i in range(size_loop + 1):
            if i % 4 == 0:
                continue
            d = lorentzian_distance(feat_mat[bar], feat_mat[i])
            if d >= last_distance:
                last_distance = d
                neighbor_distances.append(d)
                neighbor_predictions.append(int(round(y_train[i])))
                if len(neighbor_predictions) > cfg.neighbors_count:
                    idx = int(round(cfg.neighbors_count * 3 / 4))
                    last_distance = neighbor_distances[idx]
                    neighbor_distances.pop(0)
                    neighbor_predictions.pop(0)
        predictions[bar] = float(np.sum(neighbor_predictions)) if neighbor_predictions else 0.0

    vol_f = _volatility_filter(out, cfg.use_volatility_filter)
    reg_f = _regime_filter(out, cfg.regime_threshold, cfg.use_regime_filter)
    adx_f = _adx_filter(out, cfg.adx_threshold, cfg.use_adx_filter)
    f_all = vol_f & reg_f & adx_f

    filtered_signal = np.zeros(len(out), dtype=int)
    for i in range(len(out)):
        if i > 0:
            filtered_signal[i] = filtered_signal[i - 1]
        if predictions[i] > 0 and f_all.iloc[i]:
            filtered_signal[i] = 1
        elif predictions[i] < 0 and f_all.iloc[i]:
            filtered_signal[i] = -1

    close = out["close"].astype(float)
    ema_up = close > ema(close, cfg.ema_period) if cfg.use_ema_filter else pd.Series(True, index=out.index)
    ema_dn = close < ema(close, cfg.ema_period) if cfg.use_ema_filter else pd.Series(True, index=out.index)
    sma_up = close > sma(close, cfg.sma_period) if cfg.use_sma_filter else pd.Series(True, index=out.index)
    sma_dn = close < sma(close, cfg.sma_period) if cfg.use_sma_filter else pd.Series(True, index=out.index)

    fs = pd.Series(filtered_signal, index=out.index)
    is_diff = fs.ne(fs.shift(1))
    is_buy = (fs == 1) & ema_up & sma_up
    is_sell = (fs == -1) & ema_dn & sma_dn
    is_new_buy = is_buy & is_diff
    is_new_sell = is_sell & is_diff

    yhat1 = rational_quadratic(src, cfg.kernel_lookback, cfg.kernel_relative_weight, cfg.kernel_regression_level)
    yhat2 = gaussian(src, max(1, cfg.kernel_lookback - cfg.kernel_lag), cfg.kernel_regression_level)
    was_bear = yhat1.shift(2) > yhat1.shift(1)
    was_bull = yhat1.shift(2) < yhat1.shift(1)
    is_bear = yhat1.shift(1) > yhat1
    is_bull = yhat1.shift(1) < yhat1
    is_bear_change = is_bear & was_bull
    is_bull_change = is_bull & was_bear

    is_bullish = (yhat2 >= yhat1) if cfg.use_kernel_smoothing else is_bull
    is_bearish = (yhat2 <= yhat1) if cfg.use_kernel_smoothing else is_bear
    if not cfg.use_kernel_filter:
        is_bullish = pd.Series(True, index=out.index)
        is_bearish = pd.Series(True, index=out.index)

    start_long = is_new_buy & is_bullish & ema_up & sma_up
    start_short = is_new_sell & is_bearish & ema_dn & sma_dn

    bars_held = is_diff.cumsum().groupby(is_diff.cumsum()).cumcount()
    is_held_four = bars_held == 4
    is_held_lt_four = (bars_held > 0) & (bars_held < 4)
    is_last_buy = (fs.shift(4) == 1) & ema_up.shift(4).fillna(True) & sma_up.shift(4).fillna(True)
    is_last_sell = (fs.shift(4) == -1) & ema_dn.shift(4).fillna(True) & sma_dn.shift(4).fillna(True)

    end_long_strict = ((is_held_four & is_last_buy) | (is_held_lt_four & is_new_sell & is_last_buy)) & start_long.shift(4).fillna(False)
    end_short_strict = ((is_held_four & is_last_sell) | (is_held_lt_four & is_new_buy & is_last_sell)) & start_short.shift(4).fillna(False)

    # Pine: ta.crossover(yhat2, yhat1) / ta.crossunder(yhat2, yhat1) for smoothing mode
    _bull_cross = ((yhat2 >= yhat1) & (yhat2.shift(1) < yhat1.shift(1)))  # crossover
    _bear_cross = ((yhat2 <= yhat1) & (yhat2.shift(1) > yhat1.shift(1)))  # crossunder
    alert_bull = _bull_cross if cfg.use_kernel_smoothing else is_bull_change
    alert_bear = _bear_cross if cfg.use_kernel_smoothing else is_bear_change
    v_short = barssince(alert_bull) > barssince(start_short)
    v_long = barssince(alert_bear) > barssince(start_long)
    end_long_dyn = is_bear_change & v_long.shift(1).fillna(False)
    end_short_dyn = is_bull_change & v_short.shift(1).fillna(False)

    dynamic_valid = (not cfg.use_ema_filter) and (not cfg.use_sma_filter) and (not cfg.use_kernel_smoothing)
    end_long = end_long_dyn if (cfg.use_dynamic_exits and dynamic_valid) else end_long_strict
    end_short = end_short_dyn if (cfg.use_dynamic_exits and dynamic_valid) else end_short_strict

    out["prediction"] = predictions
    out["signal"] = filtered_signal
    out["start_long"] = start_long
    out["start_short"] = start_short
    out["end_long"] = end_long
    out["end_short"] = end_short
    out["kernel_estimate"] = yhat1
    return out
