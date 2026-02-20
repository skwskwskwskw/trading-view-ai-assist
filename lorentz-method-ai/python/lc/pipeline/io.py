from __future__ import annotations

import pandas as pd


CORE_COLUMNS = ["time", "open", "high", "low", "close"]


def _parse_time_column(series: pd.Series) -> pd.Series:
    # Support unix epoch (seconds or milliseconds) and ISO timestamps.
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(series) else 0.0
    if numeric_ratio > 0.9:
        median_val = numeric.dropna().abs().median() if numeric.notna().any() else 0
        unit = "ms" if median_val > 1e11 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    volume_col = None
    for candidate in ("Volume", "volume"):
        if candidate in df.columns:
            volume_col = candidate
            break

    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if volume_col is None:
        missing.append("Volume|volume")
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected = CORE_COLUMNS + [volume_col]
    out = df[selected].copy()
    out["time"] = _parse_time_column(out["time"])
    out = out.rename(columns={volume_col: "volume"})

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out
