from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd

from .ann import ANNConfig, run_ann
from .features import generate_features, pine_ann_feature_columns
from .io import load_ohlcv_csv, validate_ohlcv_columns


@dataclass
class PipelineArtifacts:
    raw: pd.DataFrame
    features: pd.DataFrame
    predictions: pd.DataFrame


def run_from_dataframe(
    df: pd.DataFrame,
    ann_config: Optional[ANNConfig] = None,
) -> pd.DataFrame:
    """Run the Lorentzian Classification pipeline.

    Args:
        df: DataFrame with required columns (time, open, high, low, close, Volume).
        ann_config: Optional ANN configuration. Uses defaults if not provided.

    Returns:
        DataFrame with original columns plus prediction/signal/trade columns.

    Raises:
        ValueError: If required columns are missing.
    """
    validated = validate_ohlcv_columns(df)
    ann_config = ann_config or ANNConfig()
    feature_df = generate_features(validated)
    pred_df = run_ann(feature_df, pine_ann_feature_columns(feature_df), ann_config)
    return pred_df


def run_end_to_end(
    data: Union[str, pd.DataFrame],
    ann_config: Optional[ANNConfig] = None,
) -> PipelineArtifacts:
    """Run the full pipeline from CSV path or DataFrame.

    Args:
        data: Either a CSV file path (str) or a DataFrame with OHLCV columns.
        ann_config: Optional ANN configuration. Uses defaults if not provided.

    Returns:
        PipelineArtifacts with raw, features, and predictions DataFrames.

    Raises:
        ValueError: If required columns are missing from the DataFrame.
    """
    ann_config = ann_config or ANNConfig()
    if isinstance(data, str):
        raw = load_ohlcv_csv(data)
    else:
        raw = validate_ohlcv_columns(data)
    feature_df = generate_features(raw)
    pred_df = run_ann(feature_df, pine_ann_feature_columns(feature_df), ann_config)
    return PipelineArtifacts(raw=raw, features=feature_df, predictions=pred_df)
