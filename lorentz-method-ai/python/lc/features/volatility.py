from __future__ import annotations

import pandas as pd

from ..ta_primitives import atr
from .base import Indicator
from .registry import register_indicator


@register_indicator
class ATRIndicator(Indicator):
    name = "atr"

    def compute(self, df: pd.DataFrame, length: int) -> pd.DataFrame:
        out = atr(df["high"].astype(float), df["low"].astype(float), df["close"].astype(float), length)
        return pd.DataFrame({f"atr_{length}": out})
