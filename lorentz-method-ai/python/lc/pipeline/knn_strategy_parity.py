from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..ta_primitives import atr, ema, rescale, rma


@dataclass
class KNNParityConfig:
    k_neighbours: int = 7
    train_size: int = 300
    pred_threshold: float = 0.0
    atr_len: int = 14
    atr_stop_mult: float = 1.5
    atr_target_mult: float = 3.0
    use_trail: bool = True
    normalize_deriv_len: int = 14
    rsi_len: int = 14
    rsi_smooth_len: int = 1
    mom_len: int = 10
    gauss_len: int = 21
    initial_equity: float = 10000.0


def _rsi(close: pd.Series, length: int) -> pd.Series:
    """Pine-compatible RSI using Wilder's smoothing (ta.rma)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _normalize_deriv(src: pd.Series, length: int) -> pd.Series:
    deriv = src - src.shift(2)
    qmean = np.sqrt((deriv.pow(2)).rolling(length).sum() / length)
    return deriv / qmean.replace(0, np.nan)


def _n_rsi(src: pd.Series, length: int, smooth_len: int) -> pd.Series:
    return rescale(ema(_rsi(src, length), smooth_len), 0, 100, 0, 1)


def _n_mom(src: pd.Series, length: int) -> pd.Series:
    mom = src - src.shift(length)
    hist_min = mom.cummin()
    hist_max = mom.cummax()
    denom = (hist_max - hist_min).replace(0, np.nan)
    return (mom - hist_min) / denom


def _gaussian_smooth(src: pd.Series, length: int, start_at_bar: int = 25) -> pd.Series:
    """Pine-compatible Gaussian smoothing with fixed-window iteration.

    Pine: ``_size = array.size(array.from(_src))`` always equals 1,
    so loop runs ``for i = 0 to _size + startAtBar`` = ``start_at_bar + 2`` iterations.
    """
    vals = src.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    pine_loop_len = 1 + start_at_bar + 1  # inclusive upper bound
    for t in range(len(vals)):
        w_sum = 0.0
        v_sum = 0.0
        max_i = min(pine_loop_len, t + 1)
        for i in range(max_i):
            w = float(np.exp(-((i**2) / (2 * (length**2)))))
            v_sum += vals[t - i] * w
            w_sum += w
        out[t] = v_sum / w_sum if w_sum else np.nan
    return pd.Series(out, index=src.index)


def _ann_knn_predict(current: np.ndarray, feats: list[np.ndarray], labels: list[float], k: int) -> float:
    dists = []
    for idx, fv in enumerate(feats):
        d = float(np.log1p(np.abs(current - fv)).sum())
        dists.append((d, labels[idx]))
    if not dists:
        return np.nan
    dists.sort(key=lambda x: x[0])
    return float(np.sum([lbl for _, lbl in dists[:k]]))


def _fill_price_for_level(is_long: bool, bar_open: float, level: float, is_stop: bool) -> float:
    if is_long:
        return min(bar_open, level) if is_stop else max(bar_open, level)
    return max(bar_open, level) if is_stop else min(bar_open, level)


def run_knn_parity(df: pd.DataFrame, cfg: KNNParityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    opn = out["open"].astype(float)

    out["norm_der"] = _normalize_deriv(close, cfg.normalize_deriv_len)
    out["rsi_n"] = _n_rsi(close, cfg.rsi_len, cfg.rsi_smooth_len)
    out["mom_n"] = _n_mom(close, cfg.mom_len)
    out["gauss_sm"] = _gaussian_smooth(close, cfg.gauss_len)
    out["atr"] = atr(high, low, close, cfg.atr_len)
    out["atr_ratio"] = out["atr"] / close.replace(0, np.nan)

    close_vals = out["close"].to_numpy()
    high_vals = out["high"].to_numpy()
    low_vals = out["low"].to_numpy()
    opn_vals = out["open"].to_numpy()
    atr_vals = out["atr"].to_numpy()

    norm_der_vals = out["norm_der"].to_numpy()
    rsi_n_vals = out["rsi_n"].to_numpy()
    mom_n_vals = out["mom_n"].to_numpy()
    gauss_sm_vals = out["gauss_sm"].to_numpy()
    atr_ratio_vals = out["atr_ratio"].to_numpy()

    feat_hist: list[np.ndarray] = []
    class_hist: list[float] = []
    prediction = np.full(len(out), np.nan)

    position = 0
    entry_price = np.nan
    entry_time: Any = None
    stop_price = np.nan
    tp_price = np.nan
    trail_price = np.nan
    trailing_offset = np.nan

    traces: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for i in range(len(out)):
        fvec = np.array([norm_der_vals[i], rsi_n_vals[i], mom_n_vals[i], gauss_sm_vals[i], atr_ratio_vals[i]], dtype=float)

        if np.isfinite(fvec).all() and len(feat_hist) >= cfg.k_neighbours:
            prediction[i] = _ann_knn_predict(fvec, feat_hist, class_hist, cfg.k_neighbours)

        pred = prediction[i]
        long_ok = position == 0 and np.isfinite(pred) and pred > cfg.pred_threshold
        short_ok = position == 0 and np.isfinite(pred) and pred < -cfg.pred_threshold

        entry_long = False
        entry_short = False
        exit_long = False
        exit_short = False
        stop_hit = False
        tp_hit = False
        trail_hit = False
        exit_reason = None

        if position != 0:
            is_long = position > 0
            h = high_vals[i]
            l = low_vals[i]
            o = opn_vals[i]

            if cfg.use_trail and np.isfinite(trailing_offset):
                if is_long:
                    if np.isnan(trail_price):
                        activation = entry_price + trailing_offset
                        if h >= activation:
                            trail_price = h - trailing_offset
                    else:
                        trail_price = max(trail_price, h - trailing_offset)
                else:
                    if np.isnan(trail_price):
                        activation = entry_price - trailing_offset
                        if l <= activation:
                            trail_price = l + trailing_offset
                    else:
                        trail_price = min(trail_price, l + trailing_offset)

            effective_stop = stop_price
            if cfg.use_trail and np.isfinite(trail_price):
                effective_stop = max(stop_price, trail_price) if is_long else min(stop_price, trail_price)

            hit_stop = (l <= effective_stop) if is_long else (h >= effective_stop)
            hit_tp = (h >= tp_price) if is_long else (l <= tp_price)

            if hit_stop or hit_tp:
                if hit_stop and hit_tp:
                    # deterministic OHLC tie-break approximation for same-bar dual hit
                    d_stop = abs(o - effective_stop)
                    d_tp = abs(o - tp_price)
                    if d_stop <= d_tp:
                        hit_tp = False
                    else:
                        hit_stop = False

                level = effective_stop if hit_stop else tp_price
                px = _fill_price_for_level(is_long=is_long, bar_open=o, level=level, is_stop=bool(hit_stop))
                if is_long:
                    exit_long = True
                else:
                    exit_short = True
                stop_hit = bool(hit_stop)
                tp_hit = bool(hit_tp)
                trail_hit = bool(hit_stop and cfg.use_trail and np.isfinite(trail_price) and np.isclose(level, trail_price))
                exit_reason = "trail" if trail_hit else ("sl" if hit_stop else "tp")
                trades.append(
                    {
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_time": out.index[i],
                        "exit_price": px,
                        "direction": "long" if is_long else "short",
                        "reason": exit_reason,
                    }
                )
                position = 0
                entry_price = np.nan
                entry_time = None
                stop_price = np.nan
                tp_price = np.nan
                trail_price = np.nan
                trailing_offset = np.nan

        # strategy.entry at close fills next bar open when process_orders_on_close = false.
        if position == 0 and i < len(out) - 1:
            if long_ok:
                position = 1
                entry_long = True
                entry_price = opn_vals[i + 1]
                entry_time = out.index[i + 1]
                # strategy.exit levels are set from current-bar close in Pine code
                stop_price = close_vals[i] - cfg.atr_stop_mult * atr_vals[i]
                tp_price = close_vals[i] + cfg.atr_target_mult * atr_vals[i]
                trailing_offset = cfg.atr_stop_mult * atr_vals[i] if cfg.use_trail else np.nan
                trail_price = np.nan
            elif short_ok:
                position = -1
                entry_short = True
                entry_price = opn_vals[i + 1]
                entry_time = out.index[i + 1]
                stop_price = close_vals[i] + cfg.atr_stop_mult * atr_vals[i]
                tp_price = close_vals[i] - cfg.atr_target_mult * atr_vals[i]
                trailing_offset = cfg.atr_stop_mult * atr_vals[i] if cfg.use_trail else np.nan
                trail_price = np.nan

        # Pine barstate.isconfirmed: append closed-bar sample AFTER calculations.
        if np.isfinite(fvec).all():
            label = 1.0 if (close_vals[i] - close_vals[i - 1] if i > 0 else 0.0) > 0 else -1.0
            if len(feat_hist) >= cfg.train_size:
                feat_hist.pop(0)
                class_hist.pop(0)
            feat_hist.append(fvec)
            class_hist.append(label)

        traces.append(
            {
                "time": out.index[i],
                "prediction": pred,
                "entry_long": entry_long,
                "exit_long": exit_long,
                "entry_short": entry_short,
                "exit_short": exit_short,
                "stop_hit": stop_hit,
                "tp_hit": tp_hit,
                "trail_hit": trail_hit,
                "exit_reason": exit_reason,
            }
        )

    trace_df = pd.DataFrame(traces).set_index("time")
    trades_df = pd.DataFrame(trades)
    return trace_df, trades_df
