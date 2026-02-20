from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .ta_primitives import atr, ema, rescale


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
    # Pine's math.sum returns na until 'length' values exist
    qmean = np.sqrt((deriv.pow(2)).rolling(length).sum() / length)
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
    # Ignore type checking on list comprehension packing to avoid slice inference issues
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

    close_vals = out["close"].to_numpy()
    high_vals = out["high"].to_numpy()
    low_vals = out["low"].to_numpy()
    opn_vals = out["open"].to_numpy()
    
    norm_der_vals = out["norm_der"].to_numpy()
    rsi_n_vals = out["rsi_n"].to_numpy()
    mom_n_vals = out["mom_n"].to_numpy()
    gauss_sm_vals = out["gauss_sm"].to_numpy()
    atr_ratio_vals = out["atr_ratio"].to_numpy()
    atr_vals = out["atr"].to_numpy()

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
        fvec = np.array([norm_der_vals[i], rsi_n_vals[i], mom_n_vals[i], gauss_sm_vals[i], atr_ratio_vals[i]], dtype=float)

        # Pine Script lookahead flaw: barstate.isconfirmed array.push happens before prediction step!
        # This gives the model access to the current bar's label at 0 distance.
        if np.isfinite(fvec).all():
            label = 1.0 if (close_vals[i] - close_vals[i-1] if i > 0 else 0) > 0 else -1.0
            if len(feat_hist) >= cfg.train_size:
                feat_hist.pop(0)
                class_hist.pop(0)
            feat_hist.append(fvec)
            class_hist.append(label)

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

        # Simulate resting exits first (intrabar)
        if position != 0:
            h = high_vals[i]
            l = low_vals[i]
            
            # Pine trailing stop (with trail_points only, offset defaults to trail_points if omitted, or sometimes it's undefined. 
            # In Pine strategy.exit, if trail_offset is unspecified it trails by trail_points distance.
            # Activation happens when profit >= trail_points.
            if position > 0:
                # LONG EXIT logic
                if cfg.use_trail:
                    # trail_points in ticks. Activation price:
                    trail_activation = entry_price + cfg.atr_stop_mult * atr_vals[i]
                    if h >= trail_activation:
                        if pd.isna(trail_anchor):
                            trail_anchor = h
                        else:
                            trail_anchor = max(trail_anchor, h)
                        # The trailing stop price is anchor - offset. Assuming offset = trail_points = 1.5 * atr
                        trail_price = trail_anchor - cfg.atr_stop_mult * atr_vals[i]

                effective_stop = stop_price
                if cfg.use_trail and not pd.isna(trail_price):
                   effective_stop = np.nanmax([stop_price, trail_price])
                   
                hit_stop = np.isfinite(effective_stop) and l <= effective_stop
                hit_tp = np.isfinite(tp_price) and h >= tp_price
                
                if hit_stop or hit_tp:
                    if hit_stop and hit_tp:
                        d_stop = abs(opn_vals[i] - effective_stop)
                        d_tp = abs(opn_vals[i] - tp_price)
                        if d_stop <= d_tp:
                            hit_tp = False
                        else:
                            hit_stop = False
                    
                    # Fill price depends on gap open
                    if hit_stop:
                        px = min(opn_vals[i], effective_stop)
                    else:
                        px = max(opn_vals[i], tp_price)
                        
                    exit_long = True
                    stop_hit = bool(hit_stop)
                    trail_hit = bool(hit_stop and cfg.use_trail and np.isfinite(trail_price) and effective_stop == trail_price)
                    tp_hit = bool(hit_tp)
                    exit_reason = "trail" if trail_hit else ("sl" if hit_stop else "tp")
                    trades.append({"entry_time": entry_time, "entry_price": entry_price, "exit_time": out.index[i], "exit_price": px, "direction": "long", "reason": exit_reason})
                    position = 0
                    
            else:
                # SHORT EXIT logic
                if cfg.use_trail:
                    # trail_points in ticks. Activation price:
                    trail_activation = entry_price - cfg.atr_stop_mult * atr_vals[i]
                    if l <= trail_activation:
                        if pd.isna(trail_anchor):
                            trail_anchor = l
                        else:
                            trail_anchor = min(trail_anchor, l)
                        # The trailing stop price is anchor + offset
                        trail_price = trail_anchor + cfg.atr_stop_mult * atr_vals[i]

                effective_stop = stop_price
                if cfg.use_trail and not pd.isna(trail_price):
                   effective_stop = np.nanmin([stop_price, trail_price])
                   
                hit_stop = np.isfinite(effective_stop) and h >= effective_stop
                hit_tp = np.isfinite(tp_price) and l <= tp_price
                
                if hit_stop or hit_tp:
                    if hit_stop and hit_tp:
                        d_stop = abs(opn_vals[i] - effective_stop)
                        d_tp = abs(opn_vals[i] - tp_price)
                        if d_stop <= d_tp:
                            hit_tp = False
                        else:
                            hit_stop = False
                            
                    # Fill price depends on gap open
                    if hit_stop:
                        px = max(opn_vals[i], effective_stop)
                    else:
                        px = min(opn_vals[i], tp_price)
                        
                    exit_short = True
                    stop_hit = bool(hit_stop)
                    trail_hit = bool(hit_stop and cfg.use_trail and np.isfinite(trail_price) and effective_stop == trail_price)
                    tp_hit = bool(hit_tp)
                    exit_reason = "trail" if trail_hit else ("sl" if hit_stop else "tp")
                    trades.append({"entry_time": entry_time, "entry_price": entry_price, "exit_time": out.index[i], "exit_price": px, "direction": "short", "reason": exit_reason})
                    position = 0

        # Simulate next entries on close (actually filled at next open, but we record decision state here).
        # We assume if the condition triggers on bar i, the actual fill is opn.iloc[i+1].
        # If we are at the last bar, we can't fill.
        if position == 0 and i < len(out) - 1:
            if long_ok:
                position = 1
                entry_long = True
                # Execution happens at the OPEN of the NEXT bar
                entry_price = opn_vals[i + 1]
                entry_time = out.index[i + 1]
                # Stops are calculated relative to entry price in Pine when using strategy.exit
                stop_price = entry_price - cfg.atr_stop_mult * atr_vals[i]
                tp_price = entry_price + cfg.atr_target_mult * atr_vals[i]
                trail_anchor = np.nan
                trail_price = np.nan
            elif short_ok:
                position = -1
                entry_short = True
                entry_price = opn_vals[i + 1]
                entry_time = out.index[i + 1]
                stop_price = entry_price + cfg.atr_stop_mult * atr_vals[i]
                tp_price = entry_price - cfg.atr_target_mult * atr_vals[i]
                trail_anchor = np.nan
                trail_price = np.nan

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
