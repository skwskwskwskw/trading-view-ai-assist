from __future__ import annotations

import pandas as pd

from ..ta_primitives import ema, sma
from .base import Indicator
from .registry import register_indicator


@register_indicator
class EMAIndicator(Indicator):
    name = "ema"

    def compute(self, df: pd.DataFrame, length: int, source: str = "close") -> pd.DataFrame:
        return pd.DataFrame({f"ema_{length}": ema(df[source].astype(float), length)})


@register_indicator
class SMAIndicator(Indicator):
    name = "sma"

    def compute(self, df: pd.DataFrame, length: int, source: str = "close") -> pd.DataFrame:
        return pd.DataFrame({f"sma_{length}": sma(df[source].astype(float), length)})
