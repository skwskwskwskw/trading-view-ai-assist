from __future__ import annotations

import numpy as np
import pandas as pd

from lc.pipeline.features import generate_features


def _sample_df(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    close = pd.Series(np.linspace(100, 120, rows))
    return pd.DataFrame(
        {
            "time": idx,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.arange(rows) + 100,
        }
    )


def test_feature_columns_exist() -> None:
    out = generate_features(_sample_df())
    for col in ["rsi_14_1", "wt_10_11", "cci_20_1", "adx_20_2", "rsi_9_1", "atr_14", "ema_200", "sma_200"]:
        assert col in out.columns


def test_no_lookahead_for_ema() -> None:
    df = _sample_df()
    out = generate_features(df)
    # shift future close should not alter historical EMA row 20
    df2 = df.copy()
    df2.loc[df2.index[-1], "close"] = df2.loc[df2.index[-1], "close"] * 2
    out2 = generate_features(df2)
    assert out.loc[20, "ema_200"] == out2.loc[20, "ema_200"]
