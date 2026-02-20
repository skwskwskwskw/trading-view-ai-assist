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


def _compare_trade_ledger(py: pd.DataFrame, pine: pd.DataFrame) -> pd.DataFrame:
    cols = ["entry_time", "exit_time", "direction"]
    left = py.copy()
    right = pine.copy()
    for c in ("entry_time", "exit_time"):
        left[c] = pd.to_datetime(left[c], utc=True, errors="coerce")
        right[c] = pd.to_datetime(right[c], utc=True, errors="coerce")

    diffs = []
    n = max(len(left), len(right))
    for i in range(n):
        if i >= len(left):
            diffs.append({"trade_index": i, "column": "missing_python", "python": None, "pine": "present"})
            continue
        if i >= len(right):
            diffs.append({"trade_index": i, "column": "missing_pine", "python": "present", "pine": None})
            continue
        for c in cols:
            pv = left.iloc[i][c]
            tv = right.iloc[i][c]
            if str(pv) != str(tv):
                diffs.append({"trade_index": i, "column": c, "python": pv, "pine": tv})
                break
    return pd.DataFrame(diffs)


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
        trade_diff = _compare_trade_ledger(ledger, pine_trades)
        trade_diff_path = outdir / "trade_mismatch.csv"
        trade_diff.to_csv(trade_diff_path, index=False)
        if trade_diff.empty:
            print("Trade event parity: PASS")
        else:
            first = trade_diff.iloc[0]
            print(
                "Trade event parity: FAIL. "
                f"First divergence trade #{first['trade_index']} "
                f"col={first['column']} py={first['python']} pine={first['pine']}"
            )
            print(f"Mismatch report: {trade_diff_path}")


if __name__ == "__main__":
    main()
