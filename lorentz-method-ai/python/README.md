# Python Lorentzian Feature + ANN Pipeline

This folder contains a modular conversion of Pine Script TA logic into Python.

## Run end-to-end

```bash
python main.py --csv "MYX_FCPO1!, 5.csv" --out predictions.csv
```

Input CSV must include exactly these columns:

- `time` (unix seconds/milliseconds or ISO string)
- `open`
- `high`
- `low`
- `close`
- `Volume` (or `volume`)

## Architecture

- `lc/features/`: pluggable indicator modules
- `lc/features/registry.py`: plugin registry + specs
- `lc/pipeline/features.py`: feature orchestration
- `lc/pipeline/ann.py`: ANN inference/training-parity logic from Pine
- `lc/pipeline/runner.py`: end-to-end pipeline entry

## Add a new indicator

1. Create a class in `lc/features/<family>.py` inheriting `Indicator`.
2. Decorate with `@register_indicator` and set unique `name`.
3. Implement `compute(df, **params) -> pd.DataFrame`.
4. Add usage spec in `lc/pipeline/features.py` using `make_spec("your_name", ...)`.

Column names must remain deterministic (e.g., `ema_20`, `rsi_14`, etc.).

## Tests

```bash
pytest -q
```

Tests cover feature shape and no-lookahead sanity checks.

## Pine to Python mapping

- `KernelFunctions.ps`:
  - `rationalQuadratic` -> `lc/features/kernel.py::rational_quadratic`
  - `gaussian` -> `lc/features/kernel.py::gaussian`
- `MLExtensions.ps`:
  - `n_rsi`, `n_cci`, `n_wt`, `n_adx` -> `lc/features/momentum.py`
  - `filter_volatility`, `filter_adx`, `regime_filter` -> `lc/pipeline/ann.py`
- `Machine Learning Lorentzian Classification.ps`:
  - Feature setup and ANN loop -> `lc/pipeline/features.py` + `lc/pipeline/ann.py`
  - Signal/entry/exit logic -> `lc/pipeline/ann.py`
- `Backtest Adapter.ps`:
  - Stream-compatible outputs (`start_long`, `end_long`, `start_short`, `end_short`) -> `lc/pipeline/ann.py`


## Trade outcome metrics (no fees)

Given pipeline output columns (`close`, `start_long`, `end_long`, `start_short`, `end_short`), you can compute per-trade outcomes and summary metrics:

```python
from lc.pipeline.trade_metrics import evaluate_trades_with_metrics

trades, summary = evaluate_trades_with_metrics(predictions_df)
print(trades[["side", "entry_price", "exit_price", "pnl"]])
print(summary)
```

Returned summary fields:
- `total_pnl`: total P&L across all closed trades.
- `max_drawdown`: maximum peak-to-trough loss on cumulative trade P&L.
- `win_rate_pct`: percentage of winning trades.
- `win_to_lose_ratio`: winners / losers (`inf` when there are winners and no losses).

