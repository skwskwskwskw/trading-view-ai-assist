from __future__ import annotations

import argparse

from lc.pipeline.runner import run_end_to_end


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Lorentzian feature engineering + ANN pipeline")
    p.add_argument("--csv", required=True, help="Path to input OHLCV CSV")
    p.add_argument("--out", default="predictions.csv", help="Where to save pipeline output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_end_to_end(args.csv)
    artifacts.predictions.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
