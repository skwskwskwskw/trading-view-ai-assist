"""Python translation of the TradingView Pine Script:
Machine Learning: Lorentzian Classification.

The implementation mirrors the original logic, including feature engineering,
Lorentzian distance ANN search, filters, kernel regression, and entry/exit rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class Settings:
    source: str = "close"
    neighbors_count: int = 8
    max_bars_back: int = 2000
    feature_count: int = 5
    color_compression: int = 1
    show_exits: bool = False
    use_dynamic_exits: bool = False


@dataclass
class FilterSettings:
    use_volatility_filter: bool = True
    use_regime_filter: bool = True
    use_adx_filter: bool = False
    regime_threshold: float = -0.1
    adx_threshold: int = 20


@dataclass
class TrendFilterSettings:
    use_ema_filter: bool = False
    ema_period: int = 200
    use_sma_filter: bool = False
    sma_period: int = 200


@dataclass
class KernelSettings:
    use_kernel_filter: bool = True
    use_kernel_smoothing: bool = False
    lookback_window: int = 8
    relative_weighting: float = 8.0
    regression_level: int = 25
    lag: int = 2


@dataclass
class FeatureSpec:
    name: str
    param_a: int
    param_b: int = 1


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


def normalize(series: pd.Series, min_val: float, max_val: float) -> pd.Series:
    hist_min = series.cummin()
    hist_max = series.cummax()
    denom = (hist_max - hist_min).replace(0, 1e-10)
    return min_val + (max_val - min_val) * (series - hist_min) / denom


def rescale(series: pd.Series, old_min: float, old_max: float, new_min: float, new_max: float) -> pd.Series:
    denom = (old_max - old_min) if (old_max - old_min) != 0 else 1e-10
    return new_min + (new_max - new_min) * (series - old_min) / denom


def n_rsi(series: pd.Series, length: int, smoothing: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rescale(ema(rsi, smoothing), 0, 100, 0, 1)


def n_cci(series: pd.Series, length: int, smoothing: int) -> pd.Series:
    tp = series
    ma = sma(tp, length)
    mad = (tp - ma).abs().rolling(window=length, min_periods=1).mean()
    cci = (tp - ma) / (0.015 * mad.replace(0, np.nan))
    return normalize(ema(cci, smoothing), 0, 1)


def n_wt(series: pd.Series, n1: int, n2: int) -> pd.Series:
    ema1 = ema(series, n1)
    ema2 = ema((series - ema1).abs(), n1)
    ci = (series - ema1) / (0.015 * ema2.replace(0, np.nan))
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, 4)
    return normalize(wt1 - wt2, 0, 1)


def n_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    plus_dm = (high - prev_high).where((high - prev_high) > (prev_low - low), 0.0).clip(lower=0)
    minus_dm = (prev_low - low).where((prev_low - low) > (high - prev_high), 0.0).clip(lower=0)

    tr_smooth = rma(tr, length)
    plus_dm_smooth = rma(plus_dm, length)
    minus_dm_smooth = rma(minus_dm, length)

    di_plus = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    di_minus = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)
    dx = (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan) * 100
    adx = rma(dx, length)
    return rescale(adx, 0, 100, 0, 1)


def barssince(condition: pd.Series) -> pd.Series:
    condition = condition.fillna(False)
    count = np.full(len(condition), np.nan)
    last_true = None
    for idx, val in enumerate(condition):
        if bool(val):
            last_true = idx
            count[idx] = 0
        elif last_true is None:
            count[idx] = np.nan
        else:
            count[idx] = idx - last_true
    return pd.Series(count, index=condition.index)


def filter_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int, threshold: int, use_filter: bool) -> pd.Series:
    if not use_filter:
        return pd.Series(True, index=close.index)
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    plus_dm = (high - prev_high).where((high - prev_high) > (prev_low - low), 0.0).clip(lower=0)
    minus_dm = (prev_low - low).where((prev_low - low) > (high - prev_high), 0.0).clip(lower=0)
    tr_smooth = rma(tr, length)
    plus_dm_smooth = rma(plus_dm, length)
    minus_dm_smooth = rma(minus_dm, length)
    di_plus = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    di_minus = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)
    dx = (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan) * 100
    adx = rma(dx, length)
    return adx > threshold


def filter_volatility(high: pd.Series, low: pd.Series, close: pd.Series, min_length: int, max_length: int, use_filter: bool) -> pd.Series:
    if not use_filter:
        return pd.Series(True, index=close.index)
    recent = atr(high, low, close, min_length)
    historical = atr(high, low, close, max_length)
    return recent > historical


def kernel_rational_quadratic(src: pd.Series, lookback: int, relative_weight: float, start_at_bar: int) -> pd.Series:
    values = src.to_numpy()
    n = len(values)
    output = np.full(n, np.nan)
    for t in range(n):
        current_weight = 0.0
        cumulative_weight = 0.0
        max_i = min(t + start_at_bar, n - 1)
        for i in range(max_i + 1):
            y = values[t - i]
            w = (1 + (i**2) / ((lookback**2) * 2 * relative_weight)) ** (-relative_weight)
            current_weight += y * w
            cumulative_weight += w
        output[t] = current_weight / cumulative_weight if cumulative_weight != 0 else np.nan
    return pd.Series(output, index=src.index)


def kernel_gaussian(src: pd.Series, lookback: int, start_at_bar: int) -> pd.Series:
    values = src.to_numpy()
    n = len(values)
    output = np.full(n, np.nan)
    for t in range(n):
        current_weight = 0.0
        cumulative_weight = 0.0
        max_i = min(t + start_at_bar, n - 1)
        for i in range(max_i + 1):
            y = values[t - i]
            w = np.exp(-(i**2) / (2 * lookback**2))
            current_weight += y * w
            cumulative_weight += w
        output[t] = current_weight / cumulative_weight if cumulative_weight != 0 else np.nan
    return pd.Series(output, index=src.index)


def series_from(name: str, close: pd.Series, high: pd.Series, low: pd.Series, hlc3: pd.Series, param_a: int, param_b: int) -> pd.Series:
    if name == "RSI":
        return n_rsi(close, param_a, param_b)
    if name == "WT":
        return n_wt(hlc3, param_a, param_b)
    if name == "CCI":
        return n_cci(close, param_a, param_b)
    if name == "ADX":
        return n_adx(high, low, close, param_a)
    raise ValueError(f"Unsupported feature {name}")


def lorentzian_distance(current: np.ndarray, historical: np.ndarray) -> float:
    return float(np.log1p(np.abs(current - historical)).sum())


def lorentzian_classification(
    data: pd.DataFrame,
    settings: Settings = Settings(),
    filter_settings: FilterSettings = FilterSettings(),
    trend_filter_settings: TrendFilterSettings = TrendFilterSettings(),
    kernel_settings: KernelSettings = KernelSettings(),
    features: Optional[Iterable[FeatureSpec]] = None,
) -> pd.DataFrame:
    """Run the Lorentzian classifier on OHLC data.

    Args:
        data: DataFrame containing columns: open, high, low, close.
    Returns:
        DataFrame with predictions, signals, and entry/exit flags.
    """
    df = data.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    hlc3 = (high + low + close) / 3

    if features is None:
        features = [
            FeatureSpec("RSI", 14, 1),
            FeatureSpec("WT", 10, 11),
            FeatureSpec("CCI", 20, 1),
            FeatureSpec("ADX", 20, 2),
            FeatureSpec("RSI", 9, 1),
        ]

    feature_series = [
        series_from(spec.name, close, high, low, hlc3, spec.param_a, spec.param_b)
        for spec in features
    ]

    feature_matrix = np.column_stack(feature_series)

    direction_long = 1
    direction_short = -1
    direction_neutral = 0

    src = df[settings.source].astype(float)
    y_train = np.where(src.shift(4) < src, direction_short, np.where(src.shift(4) > src, direction_long, direction_neutral))
    y_train = np.nan_to_num(y_train, nan=direction_neutral)

    predictions = np.zeros(len(df))
    signals = np.full(len(df), direction_neutral)

    max_bars_back_index = max(len(df) - settings.max_bars_back, 0)

    for bar_index in range(len(df)):
        last_distance = -1.0
        size = min(settings.max_bars_back - 1, bar_index)
        size_loop = min(settings.max_bars_back - 1, size)
        pred_array: list[int] = []
        dist_array: list[float] = []

        if bar_index >= max_bars_back_index:
            current_features = feature_matrix[bar_index]
            for i in range(size_loop + 1):
                if i % 4 == 0:
                    continue
                historical_features = feature_matrix[i]
                distance = lorentzian_distance(current_features, historical_features)
                if distance >= last_distance:
                    last_distance = distance
                    dist_array.append(distance)
                    pred_array.append(int(round(y_train[i])))
                    if len(pred_array) > settings.neighbors_count:
                        idx = int(round(settings.neighbors_count * 3 / 4))
                        if dist_array:
                            last_distance = dist_array[idx]
                        dist_array.pop(0)
                        pred_array.pop(0)
            predictions[bar_index] = np.sum(pred_array) if pred_array else 0.0

        if bar_index > 0:
            signals[bar_index] = signals[bar_index - 1]
        if predictions[bar_index] > 0:
            signals[bar_index] = direction_long
        elif predictions[bar_index] < 0:
            signals[bar_index] = direction_short

    volatility_filter = filter_volatility(high, low, close, 1, 10, filter_settings.use_volatility_filter)
    adx_filter = filter_adx(high, low, close, 14, filter_settings.adx_threshold, filter_settings.use_adx_filter)

    # regime filter is defined in Pine using ohlc4; approximate with hlc3 + open
    ohlc4 = (open_ + high + low + close) / 4
    if filter_settings.use_regime_filter:
        value1 = 0.2 * (ohlc4 - ohlc4.shift(1)) + 0.8 * (ohlc4 - ohlc4.shift(1)).shift(1).fillna(0)
        value2 = 0.1 * (high - low) + 0.8 * (high - low).shift(1).fillna(0)
        omega = (value1 / value2.replace(0, np.nan)).abs()
        alpha = (-omega**2 + np.sqrt(omega**4 + 16 * omega**2)) / 8
        klmf = alpha * ohlc4 + (1 - alpha) * ohlc4.shift(1).fillna(ohlc4)
        abs_slope = (klmf - klmf.shift(1)).abs()
        avg_slope = ema(abs_slope, 200)
        normalized = (abs_slope - avg_slope) / avg_slope.replace(0, np.nan)
        regime = normalized >= filter_settings.regime_threshold
    else:
        regime = pd.Series(True, index=df.index)

    filter_all = volatility_filter & regime & adx_filter

    ema_trend_up = close > ema(close, trend_filter_settings.ema_period) if trend_filter_settings.use_ema_filter else pd.Series(True, index=df.index)
    ema_trend_down = close < ema(close, trend_filter_settings.ema_period) if trend_filter_settings.use_ema_filter else pd.Series(True, index=df.index)
    sma_trend_up = close > sma(close, trend_filter_settings.sma_period) if trend_filter_settings.use_sma_filter else pd.Series(True, index=df.index)
    sma_trend_down = close < sma(close, trend_filter_settings.sma_period) if trend_filter_settings.use_sma_filter else pd.Series(True, index=df.index)

    prediction_series = pd.Series(predictions, index=df.index)
    filtered_signal = np.full(len(df), direction_neutral)
    for idx in range(len(df)):
        if prediction_series.iloc[idx] > 0 and filter_all.iloc[idx]:
            filtered_signal[idx] = direction_long
        elif prediction_series.iloc[idx] < 0 and filter_all.iloc[idx]:
            filtered_signal[idx] = direction_short
        elif idx > 0:
            filtered_signal[idx] = filtered_signal[idx - 1]
    filtered_signal = pd.Series(filtered_signal, index=df.index)

    bars_held = filtered_signal.ne(filtered_signal.shift(1)).cumsum()
    bars_held = bars_held.groupby(bars_held).cumcount()
    is_held_four = bars_held == 4
    is_held_lt_four = (bars_held > 0) & (bars_held < 4)

    is_diff_signal = filtered_signal.ne(filtered_signal.shift(1))
    is_early_flip = is_diff_signal & (
        filtered_signal.ne(filtered_signal.shift(2))
        | filtered_signal.ne(filtered_signal.shift(3))
        | filtered_signal.ne(filtered_signal.shift(4))
    )

    is_buy = (filtered_signal == direction_long) & ema_trend_up & sma_trend_up
    is_sell = (filtered_signal == direction_short) & ema_trend_down & sma_trend_down
    is_last_buy = (filtered_signal.shift(4) == direction_long) & ema_trend_up.shift(4).fillna(True) & sma_trend_up.shift(4).fillna(True)
    is_last_sell = (filtered_signal.shift(4) == direction_short) & ema_trend_down.shift(4).fillna(True) & sma_trend_down.shift(4).fillna(True)
    is_new_buy = is_buy & is_diff_signal
    is_new_sell = is_sell & is_diff_signal

    yhat1 = kernel_rational_quadratic(src, kernel_settings.lookback_window, kernel_settings.relative_weighting, kernel_settings.regression_level)
    yhat2 = kernel_gaussian(src, max(kernel_settings.lookback_window - kernel_settings.lag, 1), kernel_settings.regression_level)

    was_bear = yhat1.shift(2) > yhat1.shift(1)
    was_bull = yhat1.shift(2) < yhat1.shift(1)
    is_bear = yhat1.shift(1) > yhat1
    is_bull = yhat1.shift(1) < yhat1
    is_bear_change = is_bear & was_bull
    is_bull_change = is_bull & was_bear

    is_bull_cross = yhat2 > yhat1
    is_bear_cross = yhat2 < yhat1
    is_bull_smooth = yhat2 >= yhat1
    is_bear_smooth = yhat2 <= yhat1

    alert_bull = is_bull_cross if kernel_settings.use_kernel_smoothing else is_bull_change
    alert_bear = is_bear_cross if kernel_settings.use_kernel_smoothing else is_bear_change

    is_bullish = (is_bull_smooth if kernel_settings.use_kernel_smoothing else is_bull) if kernel_settings.use_kernel_filter else pd.Series(True, index=df.index)
    is_bearish = (is_bear_smooth if kernel_settings.use_kernel_smoothing else is_bear) if kernel_settings.use_kernel_filter else pd.Series(True, index=df.index)

    start_long = is_new_buy & is_bullish & ema_trend_up & sma_trend_up
    start_short = is_new_sell & is_bearish & ema_trend_down & sma_trend_down

    bars_since_red_entry = barssince(start_short)
    bars_since_red_exit = barssince(alert_bull)
    bars_since_green_entry = barssince(start_long)
    bars_since_green_exit = barssince(alert_bear)

    is_valid_short_exit = bars_since_red_exit > bars_since_red_entry
    is_valid_long_exit = bars_since_green_exit > bars_since_green_entry
    end_long_dynamic = is_bear_change & is_valid_long_exit.shift(1).fillna(False)
    end_short_dynamic = is_bull_change & is_valid_short_exit.shift(1).fillna(False)

    end_long_strict = ((is_held_four & is_last_buy) | (is_held_lt_four & is_new_sell & is_last_buy)) & start_long.shift(4).fillna(False)
    end_short_strict = ((is_held_four & is_last_sell) | (is_held_lt_four & is_new_buy & is_last_sell)) & start_short.shift(4).fillna(False)

    dynamic_exit_valid = not trend_filter_settings.use_ema_filter and not trend_filter_settings.use_sma_filter and not kernel_settings.use_kernel_smoothing
    if settings.use_dynamic_exits and dynamic_exit_valid:
        end_long = end_long_dynamic
        end_short = end_short_dynamic
    else:
        end_long = end_long_strict
        end_short = end_short_strict

    output = df.copy()
    output["prediction"] = prediction_series
    output["signal"] = filtered_signal
    output["start_long"] = start_long
    output["start_short"] = start_short
    output["end_long"] = end_long
    output["end_short"] = end_short
    output["is_early_signal_flip"] = is_early_flip
    output["kernel_estimate"] = yhat1

    return output
