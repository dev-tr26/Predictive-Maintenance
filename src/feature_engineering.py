"""
Adds, per unit, per sensor:
  - rolling mean / std over multiple windows  (captures local trend + noise)
  - exponentially weighted moving average     (recency-weighted trend)
  - first difference (rate of change)
  - cumulative degradation slope (linear fit of sensor vs cycle so far)

All operations are grouped by `unit_number` and computed without look ahead so this is safe to run identically on streaming/live data.
"""
from __future__ import annotations
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from src.preprocessing import SENSOR_COLS

ROLLING_WINDOWS = (5, 15)

def add_rolling_features(df: pd.DataFrame, sensor_cols=None, windows=ROLLING_WINDOWS) -> pd.DataFrame:
    sensor_cols = sensor_cols or [c for c in SENSOR_COLS if c in df.columns]
    df = df.sort_values(["unit_number", "time, in cycles"]).copy()
    grouped = df.groupby("unit_number", group_keys=False)
    
    for col in sensor_cols:
        for w in windows:
            df[f"{col}_rmean_{w}"] = grouped[col].apply(
                lambda s, w=w: s.rolling(window=w, min_periods=1).mean()
            )
            df[f"{col}_rstd_{w}"] = grouped[col].apply(
                lambda s,w=w: s.rolling(window=w, min_periods=1).std().fillna(0.0)
            )
            df[f"{col}_ewma"] = grouped[col].apply(
            lambda s: s.ewm(span=10, adjust=False).mean()
        )
        df[f"{col}_diff1"] = grouped[col].diff().fillna(0.0)

    return df

def add_degradation_slope(df: pd.DataFrame, sensor_cols=None) -> pd.DataFrame:
    """Causal linear-regression slope of each sensor vs cycle, computed over
    the history seen so far for that unit -- a cheap proxy for 'how fast is
    this sensor trending' that's far more stable than a raw first difference."""
    sensor_cols = sensor_cols or [c for c in SENSOR_COLS if c in df.columns]
    df = df.sort_values(["unit_number", "time, in cycles"]).copy()

    def _slope(group: pd.DataFrame) -> pd.DataFrame:
        x = group["time, in cycles"].to_numpy(dtype=float)
        out = {}
        for col in sensor_cols:
            y = group[col].to_numpy(dtype=float)
            slopes = np.zeros(len(y))
            for i in range(1, len(y)):
                # slope using all points up to i (inclusive), causal
                xs = x[: i + 1]
                ys = y[: i + 1]
                if len(xs) >= 2 and xs.std() > 0:
                    slopes[i] = np.polyfit(xs, ys, 1)[0]
            out[f"{col}_slope"] = slopes
        return pd.DataFrame(out, index=group.index)

    slope_df = df.groupby("unit_number", group_keys=False).apply(_slope)
    return pd.concat([df, slope_df], axis=1)


def build_feature_matrix(df: pd.DataFrame, sensor_cols=None, fast: bool = True) -> pd.DataFrame:
    """Full feature pipeline. `fast=True` skips the O(n^2)-per-unit slope
    calculation (fine for training on thousands of rows; for a live single-row
    scoring request, `fast` doesn't matter since a lightweight streaming
    slope is used in the inference path instead -- see api/inference.py)."""
    out = add_rolling_features(df, sensor_cols)
    if not fast:
        out = add_degradation_slope(out, sensor_cols)
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"unit_number", "time, in cycles", "RUL", "failure_within_window"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def fit_scaler(df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].fillna(0.0))
    return scaler

def transform(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df[feature_cols].fillna(0.0))
