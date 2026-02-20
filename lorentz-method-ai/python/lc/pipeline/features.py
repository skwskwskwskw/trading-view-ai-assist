from __future__ import annotations

import pandas as pd

from ..features import REGISTRY, IndicatorSpec, make_spec


def default_pine_feature_specs() -> list[IndicatorSpec]:
    return [
        make_spec("pine_feature", feature="RSI", param_a=14, param_b=1),
        make_spec("pine_feature", feature="WT", param_a=10, param_b=11),
        make_spec("pine_feature", feature="CCI", param_a=20, param_b=1),
        make_spec("pine_feature", feature="ADX", param_a=20, param_b=2),
        make_spec("pine_feature", feature="RSI", param_a=9, param_b=1),
        make_spec("kernel", lookback=8, relative_weight=8.0, regression_level=25, lag=2),
        make_spec("ema", length=200),
        make_spec("sma", length=200),
        make_spec("atr", length=14),
    ]


def generate_features(df: pd.DataFrame, specs: list[IndicatorSpec] | None = None) -> pd.DataFrame:
    specs = specs or default_pine_feature_specs()
    return REGISTRY.compute(df, specs)


def pine_ann_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        "rsi_14_1",
        "wt_10_11",
        "cci_20_1",
        "adx_20_2",
        "rsi_9_1",
    ]
