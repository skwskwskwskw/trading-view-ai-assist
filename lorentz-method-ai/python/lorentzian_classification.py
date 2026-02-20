"""Backwards-compatible wrapper around modular lc package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from lc.features import make_spec
from lc.pipeline.ann import ANNConfig, run_ann
from lc.pipeline.features import generate_features, pine_ann_feature_columns


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


def lorentzian_classification(
    data: pd.DataFrame,
    settings: Settings = Settings(),
    filter_settings: FilterSettings = FilterSettings(),
    trend_filter_settings: TrendFilterSettings = TrendFilterSettings(),
    kernel_settings: KernelSettings = KernelSettings(),
    features: Optional[Iterable[FeatureSpec]] = None,
) -> pd.DataFrame:
    specs = None
    if features is not None:
        specs = [
            make_spec("pine_feature", feature=f.name, param_a=f.param_a, param_b=f.param_b)
            for f in features
        ]
    featured = generate_features(data, specs)
    cfg = ANNConfig(
        neighbors_count=settings.neighbors_count,
        max_bars_back=settings.max_bars_back,
        source=settings.source,
        use_dynamic_exits=settings.use_dynamic_exits,
        use_volatility_filter=filter_settings.use_volatility_filter,
        use_regime_filter=filter_settings.use_regime_filter,
        use_adx_filter=filter_settings.use_adx_filter,
        regime_threshold=filter_settings.regime_threshold,
        adx_threshold=filter_settings.adx_threshold,
        use_ema_filter=trend_filter_settings.use_ema_filter,
        ema_period=trend_filter_settings.ema_period,
        use_sma_filter=trend_filter_settings.use_sma_filter,
        sma_period=trend_filter_settings.sma_period,
        use_kernel_filter=kernel_settings.use_kernel_filter,
        use_kernel_smoothing=kernel_settings.use_kernel_smoothing,
        kernel_lookback=kernel_settings.lookback_window,
        kernel_relative_weight=kernel_settings.relative_weighting,
        kernel_regression_level=kernel_settings.regression_level,
        kernel_lag=kernel_settings.lag,
    )
    feature_cols = pine_ann_feature_columns(featured)
    if features is not None:
        feature_cols = [f"{f.name.lower()}_{f.param_a}_{f.param_b}" for f in features]
    return run_ann(featured, feature_cols, cfg)
