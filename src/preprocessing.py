# preprocessing for sensor logs

from __future__ import annotations
import os
import numpy as np
import pandas as pd

N_SETTINGS = 3
N_SENSORS = 21

COLUMNS = (
    ["unit_number", "time, in cycles"] + 
    [f"op_setting_{i}" for i in range(1, N_SETTINGS + 1)] +
    [f"sensor_{i}" for i in range(1, N_SENSORS + 1)]
)

SENSOR_COLS = [f"sensor_{i}" for i in range(1, N_SENSORS + 1)]
SETTINGS_COLS = [f"op_setting_{i}" for i in range(1, N_SETTINGS + 1)] 

LOW_VARIANCE_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                         "sensor_16", "sensor_18", "sensor_19"]

def _read_space_delimited(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s", header=None,engine="python")
    df = df.iloc[:,: len(COLUMNS)]
    df.columns  = COLUMNS 
    return df 


def load_cmpass(data_dir: str, subset: str):
    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    test_path = os.path.join(data_dir, f"test_{subset}.txt")
    rul_path = os.path.join(data_dir, f"RUL_{subset}.txt")
    train_df = _read_space_delimited(train_path)
    print(train_df.columns.tolist())
    test_df = _read_space_delimited(test_path) if os.path.exists(test_path) else None
    rul_df = None
    if os.path.exists(rul_path):
        rul_df = pd.read_csv(rul_path, sep=r"\s+", header=None,names=["RUL"])
    return train_df,test_df,rul_df


# Compute Remaining Useful Life (cycles until failure) for training data where failure is implicitly the last recorded cycle of each unit.
def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit_number")["time, in cycles"].transform("max")
    df["RUL"] = max_cycle - df["time, in cycles"]
    return df 


# 1 = if unit will fail within warning window cycles from this row else 0
def add_failure_label(df: pd.DataFrame, warning_window: int = 30) -> pd.DataFrame:
    df = df.copy()
    if "RUL" not in df.columns:
        df = add_rul(df)
    df["failure_within_window"] = (df["RUL"] <= warning_window)
    return df 


# Piecewise-linear RUL target degradation is negligible early in life, so cap RUL at `cap` cycles to stop the model wasting capacity trying to regress a flat, uninformative region.
def clip_rul(df: pd.DataFrame,cap: int = 130) -> pd.DataFrame:
    df = df.copy()
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df 

def drop_low_varience_sensors(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    cols = cols or LOW_VARIANCE_SENSORS
    return df.drop(columns=[c for c in cols if c in df.columns])


# avoid time-series leakage by splitting by unit number not by row so no engine cycles leak
def train_val_split_by_unit(df: pd.DataFrame, val_fraction: float=0.2, seed : int =42):
    rng = np.random.default_rng(seed)
    units = df["unit_number"].unique()
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_fraction))
    val_units = set(units[:n_val])
    train = df[~df["unit_number"].isin(val_units)].reset_index(drop=True)
    val = df[df["unit_number"].isin(val_units)].reset_index(drop=True)
    return train, val    