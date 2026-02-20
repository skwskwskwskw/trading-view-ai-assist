from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class IndicatorSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class Indicator:
    name: str

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        raise NotImplementedError
