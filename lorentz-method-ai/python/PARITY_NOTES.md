# Pine ↔ Python Parity Notes (`Lorentzian-KNN-strategy-AI.ps`)

## Mapping table

| Pine feature / condition | Python implementation |
|---|---|
| `strategy()` defaults in `Lorentzian-KNN-strategy-AI.ps` (no pyramiding, close-based calculation) | `lc/pipeline/knn_strategy_parity.py::KNNParityConfig` + `run_knn_parity` position/state machine |
| `ml.normalizeDeriv(src)` | `lc/pipeline/knn_strategy_parity.py::_normalize_deriv` |
| `ml.n_rsi(src, 14)` | `lc/pipeline/knn_strategy_parity.py::_n_rsi` |
| `ml.n_mom(src, 10)` | `lc/pipeline/knn_strategy_parity.py::_n_mom` |
| `kern.gaussian(src, 21, 0.15)` | `lc/pipeline/knn_strategy_parity.py::_gaussian_smooth` |
| `ta.atr(atrLen)` | `lc/ta_primitives.py::atr` (used by `run_knn_parity`) |
| Rolling feature/class history via `var` arrays | `run_knn_parity` (`feat_hist`, `class_hist`) |
| `ml.ann_knn_predict(...)` | `lc/pipeline/knn_strategy_parity.py::_ann_knn_predict` |
| `longOk` / `shortOk` | `run_knn_parity` (`long_ok`, `short_ok`) |
| `strategy.entry` + `strategy.exit(stop/limit/trail_points)` | `run_knn_parity` trade state and intrabar exit checks |
| Per-bar signal trace | `parity_runner.py` → `python_signal_trace.csv` |
| Trade ledger | `parity_runner.py` → `python_trade_ledger.csv` |

## Pine assumptions used for parity

1. `calc_on_every_tick=false` and historical bars means calculations happen at bar close.
2. `barstate.isconfirmed` behavior is modeled by appending labels only after processing each bar.
3. Session filter default `0000-2359:1234567` is treated as always in session.
4. Stop/TP/trailing checks are evaluated intrabar from OHLC; if both stop and TP hit in the same bar, tie-break uses nearest level to open as deterministic approximation.
5. `ml.n_mom` and gaussian variant in imported TV libraries are not fully defined in this repository; Python uses documented approximation helpers in `knn_strategy_parity.py`.

## Parity runner

```bash
python parity_runner.py --csv "MYX_FCPO1!, 5.csv" --outdir parity_out
```

Optional Pine comparisons:

```bash
python parity_runner.py --csv "MYX_FCPO1!, 5.csv" --pine-trace pine_trace.csv --pine-trades pine_trades.csv
```

The script prints trade-count parity and exports first-difference details for per-bar trace mismatches.
