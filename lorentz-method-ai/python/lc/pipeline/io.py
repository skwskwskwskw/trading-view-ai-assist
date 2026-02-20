from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "Volume"]


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df[REQUIRED_COLUMNS].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.rename(columns={"Volume": "volume"})
    return out
