from __future__ import annotations

from lc.pipeline.ann import ANNConfig
from lc.pipeline.features import default_pine_feature_specs


def test_default_feature_specs_match_pine_defaults() -> None:
    specs = default_pine_feature_specs()
    normalized = [(s.name, s.params) for s in specs]
    assert normalized[:5] == [
        ("pine_feature", {"feature": "RSI", "param_a": 14, "param_b": 1}),
        ("pine_feature", {"feature": "WT", "param_a": 10, "param_b": 11}),
        ("pine_feature", {"feature": "CCI", "param_a": 20, "param_b": 1}),
        ("pine_feature", {"feature": "ADX", "param_a": 20, "param_b": 2}),
        ("pine_feature", {"feature": "RSI", "param_a": 9, "param_b": 1}),
    ]


def test_ann_config_matches_pine_defaults() -> None:
    cfg = ANNConfig()
    assert cfg.neighbors_count == 8
    assert cfg.max_bars_back == 2000
    assert cfg.source == "close"
    assert cfg.use_volatility_filter is True
    assert cfg.use_regime_filter is True
    assert cfg.use_adx_filter is False
    assert cfg.regime_threshold == -0.1
    assert cfg.adx_threshold == 20
    assert cfg.use_ema_filter is False
    assert cfg.ema_period == 200
    assert cfg.use_sma_filter is False
    assert cfg.sma_period == 200
    assert cfg.use_kernel_filter is True
    assert cfg.use_kernel_smoothing is False
    assert cfg.use_dynamic_exits is False
    assert cfg.kernel_lookback == 8
    assert cfg.kernel_relative_weight == 8.0
    assert cfg.kernel_regression_level == 25
    assert cfg.kernel_lag == 2
