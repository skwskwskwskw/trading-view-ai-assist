# Pine ↔ Python Parity Notes

## Original Lorentzian Classification Indicator

### Mapping Table: `Machine Learning Lorentzian Classification.ps` + libraries → Python

| Pine Function / Feature | Python File | Python Function / Logic |
|---|---|---|
| `series_from("RSI", ...)` → `ml.n_rsi(close, paramA, paramB)` | `lc/features/momentum.py` | `_n_rsi(close, param_a, param_b)` |
| `series_from("WT", ...)` → `ml.n_wt(hlc3, paramA, paramB)` | `lc/features/momentum.py` | `_n_wt(hlc3, n1, n2)` |
| `series_from("CCI", ...)` → `ml.n_cci(close, paramA, paramB)` | `lc/features/momentum.py` | `_n_cci(close, param_a, param_b)` |
| `series_from("ADX", ...)` → `ml.n_adx(high, low, close, paramA)` | `lc/features/momentum.py` | `_n_adx(high, low, close, length)` — recursive Wilder smoothing |
| `ml.rescale(src, oldMin, oldMax, newMin, newMax)` | `lc/ta_primitives.py` | `rescale(src, old_min, old_max, new_min, new_max)` |
| `ml.normalize(src, min, max)` (var historicMin/Max) | `lc/ta_primitives.py` | `normalize(src, new_min, new_max)` — NaN-safe expanding min/max |
| `kernels.rationalQuadratic(src, h, r, x)` | `lc/features/kernel.py` | `rational_quadratic(src, lookback, relative_weight, start_at_bar)` — fixed-window loop |
| `kernels.gaussian(src, h-lag, x)` | `lc/features/kernel.py` | `gaussian(src, lookback, start_at_bar)` — fixed-window loop |
| `ta.ema(close, period)` | `lc/ta_primitives.py` | `ema(series, length)` |
| `ta.sma(close, period)` | `lc/ta_primitives.py` | `sma(series, length)` |
| `ta.rma(src, length)` (SMA-initialized Wilder) | `lc/ta_primitives.py` | `rma(series, length)` |
| `ta.atr(length)` | `lc/ta_primitives.py` | `atr(high, low, close, length)` |
| `ml.filter_volatility(1, 10, useFilter)` | `lc/pipeline/ann.py` | `_volatility_filter(df, use_filter)` |
| `ml.regime_filter(ohlc4, threshold, useFilter)` | `lc/pipeline/ann.py` | `_regime_filter(df, threshold, use_filter)` |
| `ml.filter_adx(src, 14, threshold, useFilter)` | `lc/pipeline/ann.py` | `_adx_filter(df, threshold, use_filter)` — recursive Wilder smoothing |
| Core ANN loop (distances/predictions per bar) | `lc/pipeline/ann.py` | `run_ann()` — neighbor lists reset per bar |
| `y_train_series = src[4] < src[0] ? short : long` | `lc/pipeline/ann.py` | `y_train = np.where(src.shift(4) < src, -1, ...)` |
| `ta.barssince(condition)` | `lc/ta_primitives.py` | `barssince(condition)` |
| `ta.crossover(yhat2, yhat1)` / `ta.crossunder(yhat2, yhat1)` | `lc/pipeline/ann.py` | Crossover/crossunder via shift comparison |
| Entry/exit booleans (`startLongTrade`, etc.) | `lc/pipeline/ann.py` | `start_long`, `start_short`, `end_long`, `end_short` |
| `lorentzian_classification()` wrapper | `lorentzian_classification.py` | `lorentzian_classification()` |

---

## KNN Strategy (`Lorentzian-KNN-strategy-AI.ps`) → Python

### Mapping Table

| Pine Feature / Condition | Python Implementation |
|---|---|
| `strategy()` defaults (`pyramiding=0`, `calc_on_every_tick=false`) | `lc/pipeline/knn_strategy_parity.py::KNNParityConfig` + `run_knn_parity` position state machine |
| `ml.normalizeDeriv(src)` | `lc/pipeline/knn_strategy_parity.py::_normalize_deriv` |
| `ml.n_rsi(src, 14)` | `lc/pipeline/knn_strategy_parity.py::_n_rsi` — uses `rma()` for SMA-initialized Wilder smoothing |
| `ml.n_mom(src, 10)` | `lc/pipeline/knn_strategy_parity.py::_n_mom` |
| `kernels.gaussian(src, 21, 0.15)` | `lc/pipeline/knn_strategy_parity.py::_gaussian_smooth` — fixed-window loop |
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

---

## Pine Assumptions Used for Parity

### Original Indicator (Machine Learning Lorentzian Classification)
1. `indicator()` mode: all calculations run per closed bar (no realtime repainting).
2. Default inputs used unless overridden: neighbors=8, maxBarsBack=2000, featureCount=5, useVolatilityFilter=true, useRegimeFilter=true, useAdxFilter=false, useEmaFilter=false, useSmaFilter=false, useKernelFilter=true, useKernelSmoothing=false, useDynamicExits=false.
3. Kernel parameters: lookback=8, relativeWeighting=8.0, regressionLevel=25, lag=2.
4. Feature defaults: RSI(14,1), WT(10,11), CCI(20,1), ADX(20,2), RSI(9,1).
5. `distances` and `predictions` arrays are non-`var` — reset every bar.
6. Training label: `y_train = src[4] < src[0] ? short : src[4] > src[0] ? long : neutral`.
7. ANN loop skips `i % 4 == 0` for chronological spacing.
8. Pine's `nz()` treats NaN as 0 for arithmetic and substitutes fallback values.

### KNN Strategy
1. Historical-bar mode with `calc_on_every_tick=false`: decisions are made on closed bars.
2. `barstate.isconfirmed` behavior is modeled by appending feature/label history **after** prediction and order decisions on each bar.
3. Default session input `0000-2359:1234567` is treated as always in session.
4. `strategy.entry` orders placed on bar close are filled at next bar open (`process_orders_on_close=false` behavior).
5. `strategy.exit` levels are computed from current bar close (`close ± ATR*mult`), not from eventual next-bar fill price.
6. Trailing stop is modeled as: activate after price moves by `trail_points`; once active, update only in favorable direction.
7. If both stop and target are touched in one OHLC bar, Python uses deterministic tie-break by level nearest to bar open.
8. TradingView library implementations for `ml.n_mom` and `ml.ann_knn_predict` are imported externally; Python mirrors available behavior with deterministic local implementations.

---

## Parity Runner

```bash
# Original indicator parity (default settings)
python main.py --csv "MYX_FCPO1!, 5.csv" --out predictions.csv

# KNN strategy parity
python parity_runner.py --csv "MYX_FCPO1!, 5.csv" --outdir parity_out

# With Pine comparison data
python parity_runner.py --csv "MYX_FCPO1!, 5.csv" --pine-trace pine_trace.csv --pine-trades pine_trades.csv
```

The runner prints summary parity status and writes first-divergence reports for both signal trace and trade ledger.
