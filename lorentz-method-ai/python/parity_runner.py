from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lc.pipeline.io import load_ohlcv_csv
from lc.pipeline.knn_strategy_parity import KNNParityConfig, run_knn_parity


def _compare_bool_trace(py: pd.DataFrame, pine: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    merged = py[cols].copy().join(pine[cols].copy(), how="inner", lsuffix="_py", rsuffix="_pine")
    out = []
    for c in cols:
        dif = merged[f"{c}_py"].fillna(False).astype(bool) != merged[f"{c}_pine"].fillna(False).astype(bool)
        for idx in merged.index[dif]:
            out.append({"time": idx, "column": c, "python": bool(merged.loc[idx, f"{c}_py"]), "pine": bool(merged.loc[idx, f"{c}_pine"])})
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Parity runner for Lorentzian-KNN-strategy-AI.ps")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="parity_out")
    ap.add_argument("--pine-trace", help="Optional Pine-exported per-bar trace CSV with time + entry/exit columns")
    ap.add_argument("--pine-trades", help="Optional Pine-exported trade ledger CSV")
    args = ap.parse_args()

    df = load_ohlcv_csv(args.csv)
    trace, ledger = run_knn_parity(df, KNNParityConfig())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    trace_path = outdir / "python_signal_trace.csv"
    ledger_path = outdir / "python_trade_ledger.csv"
    trace.reset_index().to_csv(trace_path, index=False)
    ledger.to_csv(ledger_path, index=False)

    print(f"Wrote trace: {trace_path}")
    print(f"Wrote ledger: {ledger_path}")
    print(f"Python trade count: {len(ledger)}")

    if args.pine_trace:
        pine_trace = pd.read_csv(args.pine_trace)
        pine_trace["time"] = pd.to_datetime(pine_trace["time"], utc=True)
        pine_trace = pine_trace.set_index("time")
        py_trace = trace.copy()
        py_trace.index = pd.to_datetime(py_trace.index, utc=True)
        cols = ["entry_long", "exit_long", "entry_short", "exit_short", "stop_hit", "tp_hit", "trail_hit"]
        diff = _compare_bool_trace(py_trace, pine_trace, cols)
        diff_path = outdir / "trace_mismatch.csv"
        diff.to_csv(diff_path, index=False)
        if diff.empty:
            print("Trace parity: PASS")
        else:
            first = diff.iloc[0]
            print(f"Trace parity: FAIL. First divergence at {first['time']} col={first['column']} py={first['python']} pine={first['pine']}")
            print(f"Mismatch report: {diff_path}")

    if args.pine_trades:
        pine_trades = pd.read_csv(args.pine_trades)
        print(f"Pine trade count: {len(pine_trades)}")
        if len(pine_trades) == len(ledger):
            print("Trade count parity: PASS")
        else:
            print("Trade count parity: FAIL")


if __name__ == "__main__":
    main()
