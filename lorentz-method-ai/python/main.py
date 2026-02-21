from __future__ import annotations

import argparse

import pandas as pd

from lc.pipeline.runner import run_end_to_end, run_from_dataframe
from lc.pipeline.io import load_ohlcv_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Lorentzian feature engineering + ANN pipeline")
    p.add_argument("--csv", required=True, help="Path to input OHLCV CSV")
    p.add_argument("--out", default="predictions.csv", help="Where to save pipeline output")
    return p.parse_args()


def run_from_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV and run the full pipeline.

    Args:
        csv_path: Path to CSV file with columns (time, open, high, low, close, Volume).

    Returns:
        DataFrame with prediction, signal, and trade signal columns.
    """
    df = load_ohlcv_csv(csv_path)
    return run_from_dataframe(df)


def main() -> None:
    args = parse_args()
    result = run_from_csv(args.csv)
    result.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
