from __future__ import annotations

import numpy as np
import pandas as pd

from lc.pipeline.ann import ANNConfig, run_ann
from lc.pipeline.features import generate_features, pine_ann_feature_columns


def _sample_df(rows: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    base = np.linspace(100, 130, rows)
    wiggle = np.sin(np.arange(rows) / 10)
    close = pd.Series(base + wiggle)
    return pd.DataFrame(
        {
            "time": idx,
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.arange(rows) + 100,
        }
    )


def test_ann_output_shape() -> None:
    feat = generate_features(_sample_df())
    out = run_ann(feat, pine_ann_feature_columns(feat), ANNConfig(max_bars_back=200))
    assert len(out) == len(feat)
    for col in ["prediction", "signal", "start_long", "start_short", "end_long", "end_short"]:
        assert col in out.columns
