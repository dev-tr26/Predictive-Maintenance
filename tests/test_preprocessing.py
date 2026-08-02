import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    add_rul, add_failure_label, clip_rul, drop_low_variance_sensors,
    train_val_split_by_unit, SENSOR_COLS, SETTING_COLS,
)
from src.feature_engineering import add_rolling_features, get_feature_columns


def make_fake_unit_df(unit_id=1, n_cycles=50):
    rng = np.random.default_rng(0)
    data = {"unit_number": unit_id, "time_in_cycles": np.arange(1, n_cycles + 1)}
    for c in SETTING_COLS + SENSOR_COLS:
        data[c] = rng.normal(500, 5, size=n_cycles)
    return pd.DataFrame(data)


def make_fake_multi_unit_df(n_units=6, n_cycles=40):
    return pd.concat([make_fake_unit_df(u, n_cycles) for u in range(1, n_units + 1)], ignore_index=True)


def test_add_rul_decreasing_to_zero():
    df = add_rul(make_fake_unit_df(n_cycles=30))
    assert df["RUL"].iloc[0] == 29
    assert df["RUL"].iloc[-1] == 0
    assert (df["RUL"].diff().dropna() == -1).all()


def test_clip_rul_caps_at_value():
    df = add_rul(make_fake_unit_df(n_cycles=300))
    df = clip_rul(df, cap=130)
    assert df["RUL"].max() == 130


def test_add_failure_label_binary_and_matches_window():
    df = add_rul(make_fake_unit_df(n_cycles=50))
    df = add_failure_label(df, warning_window=10)
    assert set(df["failure_within_window"].unique()).issubset({0, 1})
    # last 11 rows (RUL 0..10) should be flagged as failure-soon
    assert df["failure_within_window"].sum() == 11


def test_drop_low_variance_sensors_removes_expected_columns():
    df = make_fake_unit_df()
    out = drop_low_variance_sensors(df)
    assert "sensor_1" not in out.columns
    assert "sensor_2" in out.columns  # sensor_2 is not in the low-variance list


def test_train_val_split_by_unit_no_row_leakage():
    df = make_fake_multi_unit_df(n_units=10)
    train, val = train_val_split_by_unit(df, val_fraction=0.3, seed=1)
    train_units = set(train["unit_number"].unique())
    val_units = set(val["unit_number"].unique())
    assert train_units.isdisjoint(val_units), "no unit should appear in both splits"
    assert len(val_units) >= 1
    assert len(train) + len(val) == len(df)


def test_rolling_features_are_causal_and_finite():
    df = make_fake_multi_unit_df(n_units=2, n_cycles=25)
    feat = add_rolling_features(df, sensor_cols=["sensor_2", "sensor_3"], windows=(5,))
    assert "sensor_2_rmean_5" in feat.columns
    assert "sensor_2_rstd_5" in feat.columns
    assert "sensor_2_ewma" in feat.columns
    assert not feat[["sensor_2_rmean_5", "sensor_2_rstd_5", "sensor_2_ewma"]].isna().any().any()
    # first row of each unit should equal the raw value (rolling window of 1 sample)
    first_rows = feat.groupby("unit_number").first()
    assert np.allclose(first_rows["sensor_2_rmean_5"], first_rows["sensor_2"])


def test_get_feature_columns_excludes_targets_and_ids():
    df = make_fake_unit_df()
    df = add_rul(df)
    df = add_failure_label(df)
    cols = get_feature_columns(df)
    assert "unit_number" not in cols
    assert "time_in_cycles" not in cols
    assert "RUL" not in cols
    assert "failure_within_window" not in cols
    assert "sensor_2" in cols
