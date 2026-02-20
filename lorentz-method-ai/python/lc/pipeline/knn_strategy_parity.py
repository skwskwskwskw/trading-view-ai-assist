from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..ta_primitives import atr, ema, rescale


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
    gauss_sigma_ratio: float = 0.15
    initial_equity: float = 10000.0


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _normalize_deriv(src: pd.Series, length: int) -> pd.Series:
    deriv = src - src.shift(2)
    qmean = np.sqrt((deriv.pow(2)).rolling(length, min_periods=1).sum() / length)
    return deriv / qmean.replace(0, np.nan)


def _n_rsi(src: pd.Series, length: int, smooth_len: int) -> pd.Series:
    return rescale(ema(_rsi(src, length), smooth_len), 0, 100, 0, 1)


def _n_mom(src: pd.Series, length: int) -> pd.Series:
    # Approximation for ml.n_mom from TradingView MLExtensions/2.
    mom = src - src.shift(length)
    hist_min = mom.cummin()
    hist_max = mom.cummax()
    denom = (hist_max - hist_min).replace(0, np.nan)
    return (mom - hist_min) / denom


def _gaussian_smooth(src: pd.Series, length: int, sigma_ratio: float) -> pd.Series:
    vals = src.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    sigma = max(length * sigma_ratio, 1e-9)
    for t in range(len(vals)):
        w_sum = 0.0
        v_sum = 0.0
        max_i = min(t, length - 1)
        for i in range(max_i + 1):
            w = float(np.exp(-((i**2) / (2 * sigma**2))))
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


def run_knn_parity(df: pd.DataFrame, cfg: KNNParityConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    opn = out["open"].astype(float)

    out["norm_der"] = _normalize_deriv(close, cfg.normalize_deriv_len)
    out["rsi_n"] = _n_rsi(close, cfg.rsi_len, cfg.rsi_smooth_len)
    out["mom_n"] = _n_mom(close, cfg.mom_len)
    out["gauss_sm"] = _gaussian_smooth(close, cfg.gauss_len, cfg.gauss_sigma_ratio)
    out["atr"] = atr(high, low, close, cfg.atr_len)
    out["atr_ratio"] = out["atr"] / close.replace(0, np.nan)

    feat_hist: list[np.ndarray] = []
    class_hist: list[float] = []
    prediction = np.full(len(out), np.nan)

    position = 0
    entry_price = np.nan
    entry_time: Any = None
    stop_price = np.nan
    tp_price = np.nan
    trail_anchor = np.nan
    trail_price = np.nan

    traces: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for i in range(len(out)):
        fvec = np.array([out["norm_der"].iloc[i], out["rsi_n"].iloc[i], out["mom_n"].iloc[i], out["gauss_sm"].iloc[i], out["atr_ratio"].iloc[i]], dtype=float)
        if np.isfinite(fvec).all() and len(feat_hist) >= cfg.k_neighbours:
            prediction[i] = _ann_knn_predict(fvec, feat_hist, class_hist, cfg.k_neighbours)

        in_sess = True
        pred = prediction[i]
        long_ok = in_sess and position == 0 and np.isfinite(pred) and pred > cfg.pred_threshold
        short_ok = in_sess and position == 0 and np.isfinite(pred) and pred < -cfg.pred_threshold

        entry_long = False
        entry_short = False
        exit_long = False
        exit_short = False
        stop_hit = False
        tp_hit = False
        trail_hit = False
        exit_reason = None

        # Simulate resting exits first (intrabar), then next entries on close.
        if position != 0:
            h = high.iloc[i]
            l = low.iloc[i]
            if position > 0:
                if cfg.use_trail:
                    trail_anchor = max(trail_anchor, h)
                    trail_price = trail_anchor - cfg.atr_stop_mult * out["atr"].iloc[i]
                effective_stop = np.nanmax([stop_price, trail_price]) if cfg.use_trail else stop_price
                hit_stop = np.isfinite(effective_stop) and l <= effective_stop
                hit_tp = np.isfinite(tp_price) and h >= tp_price
                if hit_stop or hit_tp:
                    if hit_stop and hit_tp:
                        d_stop = abs(opn.iloc[i] - effective_stop)
                        d_tp = abs(opn.iloc[i] - tp_price)
                        if d_stop <= d_tp:
                            hit_tp = False
                        else:
                            hit_stop = False
                    px = effective_stop if hit_stop else tp_price
                    exit_long = True
                    stop_hit = bool(hit_stop)
                    trail_hit = bool(hit_stop and cfg.use_trail and np.isfinite(trail_price) and effective_stop == trail_price)
                    tp_hit = bool(hit_tp)
                    exit_reason = "trail" if trail_hit else ("sl" if hit_stop else "tp")
                    trades.append({"entry_time": entry_time, "entry_price": entry_price, "exit_time": out.index[i], "exit_price": px, "direction": "long", "reason": exit_reason})
                    position = 0
            else:
                if cfg.use_trail:
                    trail_anchor = min(trail_anchor, l)
                    trail_price = trail_anchor + cfg.atr_stop_mult * out["atr"].iloc[i]
                effective_stop = np.nanmin([stop_price, trail_price]) if cfg.use_trail else stop_price
                hit_stop = np.isfinite(effective_stop) and h >= effective_stop
                hit_tp = np.isfinite(tp_price) and l <= tp_price
                if hit_stop or hit_tp:
                    if hit_stop and hit_tp:
                        d_stop = abs(opn.iloc[i] - effective_stop)
                        d_tp = abs(opn.iloc[i] - tp_price)
                        if d_stop <= d_tp:
                            hit_tp = False
                        else:
                            hit_stop = False
                    px = effective_stop if hit_stop else tp_price
                    exit_short = True
                    stop_hit = bool(hit_stop)
                    trail_hit = bool(hit_stop and cfg.use_trail and np.isfinite(trail_price) and effective_stop == trail_price)
                    tp_hit = bool(hit_tp)
                    exit_reason = "trail" if trail_hit else ("sl" if hit_stop else "tp")
                    trades.append({"entry_time": entry_time, "entry_price": entry_price, "exit_time": out.index[i], "exit_price": px, "direction": "short", "reason": exit_reason})
                    position = 0

        if position == 0:
            if long_ok:
                position = 1
                entry_long = True
                entry_price = close.iloc[i]
                entry_time = out.index[i]
                stop_price = close.iloc[i] - cfg.atr_stop_mult * out["atr"].iloc[i]
                tp_price = close.iloc[i] + cfg.atr_target_mult * out["atr"].iloc[i]
                trail_anchor = high.iloc[i]
                trail_price = trail_anchor - cfg.atr_stop_mult * out["atr"].iloc[i]
            elif short_ok:
                position = -1
                entry_short = True
                entry_price = close.iloc[i]
                entry_time = out.index[i]
                stop_price = close.iloc[i] + cfg.atr_stop_mult * out["atr"].iloc[i]
                tp_price = close.iloc[i] - cfg.atr_target_mult * out["atr"].iloc[i]
                trail_anchor = low.iloc[i]
                trail_price = trail_anchor + cfg.atr_stop_mult * out["atr"].iloc[i]

        # Pine barstate.isconfirmed: push this closed bar feature + label for future bars.
        if np.isfinite(fvec).all():
            label = 1.0 if close.diff().iloc[i] > 0 else -1.0
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
