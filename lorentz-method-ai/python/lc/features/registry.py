from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Indicator, IndicatorSpec


class IndicatorRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Indicator] = {}

    def register(self, indicator: Indicator) -> None:
        self._items[indicator.name] = indicator

    def compute(self, df: pd.DataFrame, specs: list[IndicatorSpec]) -> pd.DataFrame:
        output = df.copy()
        for spec in specs:
            if spec.name not in self._items:
                raise KeyError(f"Unknown indicator: {spec.name}")
            feats = self._items[spec.name].compute(output, **spec.params)
            output = pd.concat([output, feats], axis=1)
        return output


REGISTRY = IndicatorRegistry()


def register_indicator(cls: type[Indicator]) -> type[Indicator]:
    REGISTRY.register(cls())
    return cls


def make_spec(name: str, **params: Any) -> IndicatorSpec:
    return IndicatorSpec(name=name, params=params)
