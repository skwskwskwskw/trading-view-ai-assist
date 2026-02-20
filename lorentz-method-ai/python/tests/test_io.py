from __future__ import annotations

import pandas as pd

from lc.pipeline.io import _parse_time_column


def test_parse_time_iso() -> None:
    s = pd.Series(["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"])
    out = _parse_time_column(s)
    assert str(out.dtype).startswith("datetime64")
    assert out.notna().all()


def test_parse_time_unix_seconds() -> None:
    s = pd.Series([1704067200, 1704067500])
    out = _parse_time_column(s)
    assert out.notna().all()
    assert int(out.iloc[1].timestamp() - out.iloc[0].timestamp()) == 300


def test_parse_time_unix_milliseconds() -> None:
    s = pd.Series([1704067200000, 1704067500000])
    out = _parse_time_column(s)
    assert out.notna().all()
    assert int(out.iloc[1].timestamp() - out.iloc[0].timestamp()) == 300
