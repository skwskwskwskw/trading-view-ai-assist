"""Streamlit app for the Lorentzian Classification Python port.

Workflow:
1) Upload CSV with OHLCV columns.
2) Automatically search feature parameters to maximize profit with walk-forward evaluation.
3) Report best parameters and resulting profit without look-ahead bias.
"""
from __future__ import annotations

import itertools
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from lorentzian_classification import (
    FeatureSpec,
    FilterSettings,
    KernelSettings,
    Settings,
    TrendFilterSettings,
    lorentzian_classification,
)


@dataclass
class OptimizationResult:
    params: dict
    profit: float


DEFAULT_CSV_NAME = "MYX_FCPO1!, 5.csv"


def load_ohlcv(
    csv_file: Optional[st.runtime.uploaded_file_manager.UploadedFile] = None,
    file_path: Optional[str] = None,
) -> pd.DataFrame:
    if csv_file is None and file_path is None:
        raise ValueError("No CSV file provided.")
    df = pd.read_csv(csv_file or file_path)
    df.columns = [col.strip().lower() for col in df.columns]
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def generate_range(min_val: int, max_val: int, step: int) -> List[int]:
    return list(range(min_val, max_val + 1, step))


def generate_float_range(min_val: float, max_val: float, step: float) -> List[float]:
    values = []
    current = min_val
    while current <= max_val + 1e-9:
        values.append(round(current, 5))
        current += step
    return values


def evaluate_trades(df: pd.DataFrame) -> float:
    position = 0
    entry_price = 0.0
    profit = 0.0
    for idx, row in df.iterrows():
        if row["start_long"] and position == 0:
            position = 1
            entry_price = row["close"]
        elif row["start_short"] and position == 0:
            position = -1
            entry_price = row["close"]
        elif row["end_long"] and position == 1:
            profit += row["close"] - entry_price
            position = 0
        elif row["end_short"] and position == -1:
            profit += entry_price - row["close"]
            position = 0
    if position == 1:
        profit += df.iloc[-1]["close"] - entry_price
    elif position == -1:
        profit += entry_price - df.iloc[-1]["close"]
    return float(profit)


def run_strategy(
    data: pd.DataFrame,
    settings: Settings,
    filter_settings: FilterSettings,
    trend_filter_settings: TrendFilterSettings,
    kernel_settings: KernelSettings,
    features: Iterable[FeatureSpec],
) -> pd.DataFrame:
    return lorentzian_classification(
        data,
        settings=settings,
        filter_settings=filter_settings,
        trend_filter_settings=trend_filter_settings,
        kernel_settings=kernel_settings,
        features=features,
    )


def walk_forward_optimize(
    data: pd.DataFrame,
    settings: Settings,
    filter_settings: FilterSettings,
    trend_filter_settings: TrendFilterSettings,
    kernel_settings: KernelSettings,
    feature_grid: List[List[FeatureSpec]],
    train_window: int,
    test_window: int,
    max_combinations: Optional[int] = None,
) -> OptimizationResult:
    total_profit = 0.0
    best_params = None
    start = 0

    while start + train_window + test_window <= len(data):
        train_end = start + train_window
        test_end = train_end + test_window

        train_data = data.iloc[:train_end]
        test_data = data.iloc[:test_end]

        best_train_profit = -np.inf
        best_feature_set = None

        if max_combinations is not None and max_combinations < len(feature_grid):
            sampled_features = random.sample(feature_grid, max_combinations)
        else:
            sampled_features = feature_grid

        for features in sampled_features:
            result = run_strategy(
                train_data,
                settings,
                filter_settings,
                trend_filter_settings,
                kernel_settings,
                features,
            )
            profit = evaluate_trades(result)
            if profit > best_train_profit:
                best_train_profit = profit
                best_feature_set = features

        if best_feature_set is None:
            break

        test_result = run_strategy(
            test_data,
            settings,
            filter_settings,
            trend_filter_settings,
            kernel_settings,
            best_feature_set,
        )
        test_slice = test_result.iloc[train_end:test_end]
        total_profit += evaluate_trades(test_slice)
        best_params = {
            "features": best_feature_set,
            "train_profit": best_train_profit,
        }
        start = test_end

    return OptimizationResult(params=best_params or {}, profit=total_profit)


def build_feature_grid(feature_names: List[str], param_ranges: List[dict]) -> List[List[FeatureSpec]]:
    feature_grids = []
    for name, ranges in zip(feature_names, param_ranges):
        a_values = generate_range(ranges["param_a_min"], ranges["param_a_max"], ranges["param_a_step"])
        b_values = generate_range(ranges["param_b_min"], ranges["param_b_max"], ranges["param_b_step"])
        feature_grids.append([FeatureSpec(name, a, b) for a, b in itertools.product(a_values, b_values)])

    combos = []
    for feature_combo in itertools.product(*feature_grids):
        combos.append(list(feature_combo))
    return combos


def main() -> None:
    st.set_page_config(page_title="Lorentzian Classification", layout="wide")
    st.title("Lorentzian Classification - Walk-Forward Optimizer")

    default_path = os.path.join(os.path.dirname(__file__), DEFAULT_CSV_NAME)
    uploaded = st.file_uploader(
        "Upload OHLCV CSV (optional)",
        type=["csv"],
        help="Expected columns: time (unix), open, high, low, close, Volume.",
    )
    use_default = st.checkbox(f"Use default CSV ({DEFAULT_CSV_NAME})", value=True)
    data_source_label = "uploaded file"

    try:
        if uploaded is not None and not use_default:
            data = load_ohlcv(csv_file=uploaded)
        elif os.path.exists(default_path):
            data = load_ohlcv(file_path=default_path)
            data_source_label = DEFAULT_CSV_NAME
        elif uploaded is not None:
            data = load_ohlcv(csv_file=uploaded)
        else:
            st.error(f"Default file not found at {default_path}. Upload a CSV to continue.")
            return
    except ValueError as exc:
        st.error(str(exc))
        return

    st.sidebar.header("General Settings")
    settings = Settings(
        source=st.sidebar.selectbox("Source", ["close", "open", "high", "low"], index=0),
        neighbors_count=st.sidebar.slider("Neighbors Count", 1, 25, 8),
        max_bars_back=st.sidebar.slider("Max Bars Back", 200, 5000, 2000),
        feature_count=st.sidebar.slider("Feature Count", 1, 10, 5),
        color_compression=st.sidebar.slider("Color Compression", 1, 10, 1),
        show_exits=st.sidebar.checkbox("Show Default Exits", value=False),
        use_dynamic_exits=st.sidebar.checkbox("Use Dynamic Exits", value=False),
    )

    st.sidebar.header("Filter Settings")
    filter_settings = FilterSettings(
        use_volatility_filter=st.sidebar.checkbox("Use Volatility Filter", value=True),
        use_regime_filter=st.sidebar.checkbox("Use Regime Filter", value=True),
        use_adx_filter=st.sidebar.checkbox("Use ADX Filter", value=False),
        regime_threshold=st.sidebar.slider("Regime Threshold", -10.0, 10.0, -0.1, 0.1),
        adx_threshold=st.sidebar.slider("ADX Threshold", 0, 100, 20),
    )

    st.sidebar.header("Trend Filters")
    trend_filter_settings = TrendFilterSettings(
        use_ema_filter=st.sidebar.checkbox("Use EMA Filter", value=False),
        ema_period=st.sidebar.slider("EMA Period", 1, 400, 200),
        use_sma_filter=st.sidebar.checkbox("Use SMA Filter", value=False),
        sma_period=st.sidebar.slider("SMA Period", 1, 400, 200),
    )

    st.sidebar.header("Kernel Settings")
    kernel_settings = KernelSettings(
        use_kernel_filter=st.sidebar.checkbox("Use Kernel Filter", value=True),
        use_kernel_smoothing=st.sidebar.checkbox("Use Kernel Smoothing", value=False),
        lookback_window=st.sidebar.slider("Kernel Lookback Window", 3, 50, 8),
        relative_weighting=st.sidebar.slider("Kernel Relative Weighting", 0.25, 25.0, 8.0, 0.25),
        regression_level=st.sidebar.slider("Kernel Regression Level", 2, 25, 25),
        lag=st.sidebar.slider("Kernel Lag", 1, 2, 2),
    )

    st.sidebar.header("Walk-Forward Settings")
    train_window = st.sidebar.slider("Training Window (bars)", 200, 3000, 1000, 50)
    test_window = st.sidebar.slider("Test Window (bars)", 50, 1000, 250, 25)
    optimizer_mode = st.sidebar.selectbox("Optimizer", ["Grid Search", "Random Search"], index=0)
    max_combinations = None
    if optimizer_mode == "Random Search":
        max_combinations = st.sidebar.number_input("Random Samples per Window", 10, 5000, 200)

    st.sidebar.header("Feature Parameters (Auto-Optimized)")
    default_features = [
        ("RSI", 14, 1),
        ("WT", 10, 11),
        ("CCI", 20, 1),
        ("ADX", 20, 2),
        ("RSI", 9, 1),
        ("WT", 7, 14),
        ("CCI", 30, 2),
        ("ADX", 14, 2),
        ("RSI", 21, 1),
        ("WT", 5, 9),
    ]

    feature_names = []
    param_ranges = []
    for idx in range(settings.feature_count):
        name, default_a, default_b = default_features[idx]
        st.sidebar.subheader(f"Feature {idx + 1}")
        feature_name = st.sidebar.selectbox(
            f"Type {idx + 1}",
            ["RSI", "WT", "CCI", "ADX"],
            index=["RSI", "WT", "CCI", "ADX"].index(name),
            key=f"feature_type_{idx}",
        )
        param_a_min = st.sidebar.number_input(
            f"F{idx + 1} Param A Min",
            2,
            50,
            min(14, default_a),
            key=f"param_a_min_{idx}",
        )
        param_a_max = st.sidebar.number_input(
            f"F{idx + 1} Param A Max",
            2,
            100,
            max(14, default_a),
            key=f"param_a_max_{idx}",
        )
        param_a_step = st.sidebar.number_input(
            f"F{idx + 1} Param A Step",
            1,
            10,
            1,
            key=f"param_a_step_{idx}",
        )
        param_b_min = st.sidebar.number_input(
            f"F{idx + 1} Param B Min",
            1,
            20,
            min(1, default_b),
            key=f"param_b_min_{idx}",
        )
        param_b_max = st.sidebar.number_input(
            f"F{idx + 1} Param B Max",
            1,
            30,
            max(1, default_b),
            key=f"param_b_max_{idx}",
        )
        param_b_step = st.sidebar.number_input(
            f"F{idx + 1} Param B Step",
            1,
            10,
            1,
            key=f"param_b_step_{idx}",
        )
        feature_names.append(feature_name)
        param_ranges.append(
            {
                "param_a_min": int(param_a_min),
                "param_a_max": int(param_a_max),
                "param_a_step": int(param_a_step),
                "param_b_min": int(param_b_min),
                "param_b_max": int(param_b_max),
                "param_b_step": int(param_b_step),
            }
        )

    if st.button("Run Optimization"):
        if len(data) < train_window + test_window:
            st.error("Not enough data for the selected training + test windows.")
            return
        feature_grid = build_feature_grid(feature_names, param_ranges)
        if len(feature_grid) == 0:
            st.error("No feature combinations to test. Adjust parameter ranges.")
            return

        if max_combinations is not None:
            st.write(f"Sampling {max_combinations} combinations per window from {len(feature_grid)} total.")
        else:
            st.write(f"Testing {len(feature_grid)} feature combinations per window.")
        result = walk_forward_optimize(
            data,
            settings,
            filter_settings,
            trend_filter_settings,
            kernel_settings,
            feature_grid,
            train_window=train_window,
            test_window=test_window,
            max_combinations=max_combinations,
        )

        st.subheader("Walk-Forward Results")
        st.caption(f"Data source: {data_source_label}")
        st.metric("Total Profit (units)", f"{result.profit:.2f}")

        if result.params:
            st.write("Best parameters from the last window:")
            for idx, spec in enumerate(result.params["features"], start=1):
                st.write(f"Feature {idx}: {spec.name} (A={spec.param_a}, B={spec.param_b})")
            st.write(f"Training-window profit (last window): {result.params['train_profit']:.2f}")

        st.caption(
            "Optimization uses walk-forward windows. Each test window is evaluated only with parameters fitted "
            "on historical data preceding it, avoiding look-ahead bias."
        )


if __name__ == "__main__":
    main()
