from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Indicator
from .registry import register_indicator


def rational_quadratic(src: pd.Series, lookback: int, relative_weight: float, start_at_bar: int) -> pd.Series:
    vals = src.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for t in range(len(vals)):
        current_w = 0.0
        cumulative_w = 0.0
        max_i = min(t + start_at_bar, len(vals) - 1)
        for i in range(max_i + 1):
            y = vals[t - i]
            w = (1 + (i**2) / ((lookback**2) * 2 * relative_weight)) ** (-relative_weight)
            current_w += y * w
            cumulative_w += w
        out[t] = current_w / cumulative_w if cumulative_w else np.nan
    return pd.Series(out, index=src.index)


def gaussian(src: pd.Series, lookback: int, start_at_bar: int) -> pd.Series:
    vals = src.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for t in range(len(vals)):
        current_w = 0.0
        cumulative_w = 0.0
        max_i = min(t + start_at_bar, len(vals) - 1)
        for i in range(max_i + 1):
            y = vals[t - i]
            w = np.exp(-(i**2) / (2 * (lookback**2)))
            current_w += y * w
            cumulative_w += w
        out[t] = current_w / cumulative_w if cumulative_w else np.nan
    return pd.Series(out, index=src.index)


@register_indicator
class KernelIndicator(Indicator):
    name = "kernel"

    def compute(
        self,
        df: pd.DataFrame,
        source: str = "close",
        lookback: int = 8,
        relative_weight: float = 8.0,
        regression_level: int = 25,
        lag: int = 2,
    ) -> pd.DataFrame:
        src = df[source].astype(float)
        rq = rational_quadratic(src, lookback, relative_weight, regression_level)
        ga = gaussian(src, max(1, lookback - lag), regression_level)
        return pd.DataFrame(
            {
                f"kernel_rq_{lookback}_{relative_weight}_{regression_level}": rq,
                f"kernel_gauss_{lookback}_{lag}_{regression_level}": ga,
            }
        )
