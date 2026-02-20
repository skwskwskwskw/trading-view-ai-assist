from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .ann import ANNConfig, run_ann
from .features import generate_features, pine_ann_feature_columns
from .io import load_ohlcv_csv


@dataclass
class PipelineArtifacts:
    raw: pd.DataFrame
    features: pd.DataFrame
    predictions: pd.DataFrame


def run_end_to_end(csv_path: str, ann_config: ANNConfig | None = None) -> PipelineArtifacts:
    ann_config = ann_config or ANNConfig()
    raw = load_ohlcv_csv(csv_path)
    feature_df = generate_features(raw)
    pred_df = run_ann(feature_df, pine_ann_feature_columns(feature_df), ann_config)
    return PipelineArtifacts(raw=raw, features=feature_df, predictions=pred_df)
