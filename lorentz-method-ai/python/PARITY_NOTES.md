# Pine ↔ Python Parity Notes (`Lorentzian-KNN-strategy-AI.ps`)

## Mapping table

| Pine feature / condition | Python implementation |
|---|---|
| `strategy()` defaults (`pyramiding=0`, `calc_on_every_tick=false`) | `lc/pipeline/knn_strategy_parity.py::KNNParityConfig` + `run_knn_parity` position state machine |
| `ml.normalizeDeriv(src)` | `lc/pipeline/knn_strategy_parity.py::_normalize_deriv` |
| `ml.n_rsi(src, 14)` | `lc/pipeline/knn_strategy_parity.py::_n_rsi` |
| `ml.n_mom(src, 10)` | `lc/pipeline/knn_strategy_parity.py::_n_mom` |
| `kernels.gaussian(src, 21, 0.15)` | `lc/pipeline/knn_strategy_parity.py::_gaussian_smooth` |
| `ta.atr(atrLen)` | `lc/ta_primitives.py::atr` (consumed in `run_knn_parity`) |
| Rolling `var` arrays (`featHist`, `classHist`) | `run_knn_parity` (`feat_hist`, `class_hist`) |
| `ml.ann_knn_predict(...)` | `lc/pipeline/knn_strategy_parity.py::_ann_knn_predict` |
| `longOk` / `shortOk` | `run_knn_parity` (`long_ok`, `short_ok`) |
| `strategy.entry(...)` placement | `run_knn_parity` records entry signal at decision bar and fills at next-bar open |
| `strategy.exit(stop/limit/trail_points)` | `run_knn_parity` fixed stop/target + trailing activation/update logic |
| Per-bar signal trace | `parity_runner.py` → `python_signal_trace.csv` |
| Trade ledger | `parity_runner.py` → `python_trade_ledger.csv` |
| Pine-vs-Python trace diff | `parity_runner.py` → `trace_mismatch.csv` |
| Pine-vs-Python trade-event diff | `parity_runner.py` → `trade_mismatch.csv` |

## Pine assumptions used for parity

1. Historical-bar mode with `calc_on_every_tick=false`: decisions are made on closed bars.
2. `barstate.isconfirmed` behavior is modeled by appending feature/label history **after** prediction and order decisions on each bar.
3. Default session input `0000-2359:1234567` is treated as always in session.
4. `strategy.entry` orders placed on bar close are filled at next bar open (`process_orders_on_close=false` behavior).
5. `strategy.exit` levels in this Pine strategy are computed from current bar close (`close ± ATR*mult`), not from eventual next-bar fill price.
6. Trailing stop is modeled as: activate after price moves by `trail_points`; once active, update only in favorable direction.
7. If both stop and target are touched in one OHLC bar, Python uses deterministic tie-break by level nearest to bar open (documented approximation).
8. TradingView library implementations for `ml.n_mom` and `ml.ann_knn_predict` are imported externally; Python mirrors available behavior with deterministic local implementations.

## Parity runner

```bash
python parity_runner.py --csv "MYX_FCPO1!, 5.csv" --outdir parity_out
```

Optional Pine comparisons:

```bash
python parity_runner.py --csv "MYX_FCPO1!, 5.csv" --pine-trace pine_trace.csv --pine-trades pine_trades.csv
```

The runner prints summary parity status and writes first-divergence reports for both signal trace and trade ledger.
