# simulates real test data subset from which the model was trained on live telemetry for fleet of equipment units, each progressing through a degradation curve like the offline training data.
# Stands in for a real telemetry source (PLC/SCADA/OPC-UA/MQTT) in this demo to show in real time indashboard.
from __future__ import annotations
import numpy as np
import pandas as pd 
from src.preprocessing import SENSOR_COLS, SETTINGS_COLS

# Sensors that trend during degradation, matching data/generate_synthetic_cmapss.py
DEGRADING_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]
EQUIPMENT_TYPES = ["turbine", "compressor", "pump", "generator"]

from src.preprocessing import load_cmpass


class RealDataFleetSimulator:
    def __init__(self,data_dir: str, subset: str, n_units: int = 6, seed: int = 7):
        self.data_dir = data_dir
        self.subset = subset
        self.rng = np.random.default_rng(seed)
        
        train_df, test_df, rul_df = load_cmpass(data_dir, subset=subset)
        # Prefer the test split (held-out units, truncated before failure --
        # exactly the "unseen live equipment" scenario this demo simulates).
        # Fall back to train if a test file isn't present for this subset.
        source_df = test_df if test_df is not None and len(test_df) > 0 else train_df
        self.source_split = "test" if source_df is test_df else "train"
        
        available_units = sorted(source_df["unit_number"].unique())
        n_pick = min(n_units, len(available_units))
        chosen = self.rng.choice(available_units, size=n_pick, replace=False)        
        
        self.unit_ids = [f"unit-{i+1:02d}" for i in range(n_units)]
        self.equipment_type = {
            uid: EQUIPMENT_TYPES[i % len(EQUIPMENT_TYPES)] for i, uid in enumerate(self.unit_ids)
        }
        
        self.source_unit_number = {}   # slot -> real unit_number from the file
        self.unit_frames: dict[str, pd.DataFrame] = {}
        self.pointer: dict[str, int] = {}

        for slot, real_unit in zip(self.unit_ids, chosen):
            frame = (
                source_df[source_df["unit_number"] == real_unit]
                .sort_values("time, in cycles")
                .reset_index(drop=True)
            )
            self.unit_frames[slot] = frame
            self.source_unit_number[slot] = int(real_unit)
            self.pointer[slot] = 0

    def step(self, uid: str):
        """Return (reading_dict, cycle) for the next real recorded row for
        this slot, looping back to the start of that unit's recorded history
        once exhausted."""
        frame = self.unit_frames[uid]
        idx = self.pointer[uid]
        row = frame.iloc[idx]

        reading = {col: float(row[col]) for col in SETTINGS_COLS + SENSOR_COLS}
        cycle = int(row["time, in cycles"])

        self.pointer[uid] = (idx + 1) % len(frame)
        return reading, cycle
    
    def info(self) -> dict:
        return {
            "subset": self.subset,
            "source_split": self.source_split,
            "units": {
                uid: {"real_unit_number": self.source_unit_number[uid],
                      "recorded_cycles": len(self.unit_frames[uid])}
                for uid in self.unit_ids
            },
        }
        
class SyntheticFleetSimulator:
    """Kept as a fallback for environments with no CMAPSS-format data files
    at all. Generates synthetic degradation curves rather than replaying
    real recordings -- see RealDataFleetSimulator (default) for real data.
    """
    def __init__(self, n_units: int = 6, seed: int = 7):
        self.subset = "synthetic"
        self.source_split = "synthetic"
        self.rng = np.random.default_rng(seed)
        self.unit_ids = [f"unit-{i+1:02d}" for i in range(n_units)]
        self.equipment_type = {
            uid: EQUIPMENT_TYPES[i % len(EQUIPMENT_TYPES)] for i, uid in enumerate(self.unit_ids)
        }
        self.degrading_sensors = [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]
        self.cycle: dict[str, int] = {}
        self.max_life: dict[str, int] = {}
        self.base_levels: dict[str, np.ndarray] = {}
        self.noise_scale: dict[str, np.ndarray] = {}
        self.trend_dir: dict[str, np.ndarray] = {}
        self.trend_mag: dict[str, np.ndarray] = {}
        for uid in self.unit_ids:
            self._reset_unit(uid)

    def _reset_unit(self, uid: str):
        self.cycle[uid] = int(self.rng.integers(1, 60))
        self.max_life[uid] = int(self.rng.integers(180, 320))
        self.base_levels[uid] = self.rng.normal(500, 12, size=len(SENSOR_COLS))
        self.noise_scale[uid] = self.rng.uniform(0.3, 0.8, size=len(SENSOR_COLS))
        self.trend_dir[uid] = self.rng.choice([-1, 1], size=len(SENSOR_COLS))
        self.trend_mag[uid] = self.rng.uniform(15, 60, size=len(SENSOR_COLS))

    def step(self, uid: str):
        self.cycle[uid] += 1
        cycle = self.cycle[uid]
        life = self.max_life[uid]
        if cycle >= life:
            self._reset_unit(uid)
            cycle = self.cycle[uid]
            life = self.max_life[uid]

        degradation = (cycle / life) ** 1.6
        reading = {}
        for i, col in enumerate(SENSOR_COLS):
            sensor_idx = i + 1
            level = self.base_levels[uid][i]
            noise = self.rng.normal(0, self.noise_scale[uid][i])
            trend = 0.0
            if sensor_idx in self.degrading_sensors:
                trend = self.trend_dir[uid][i] * self.trend_mag[uid][i] * degradation
            reading[col] = float(level + trend + noise)
        for i, col in enumerate(SETTINGS_COLS):
            base = [0.0, 0.0, 100.0][i]
            scale = [0.002, 0.0003, 0.02][i]
            reading[col] = float(self.rng.normal(base, scale))
        return reading, cycle

    def info(self) -> dict:
        return {"subset": self.subset, "source_split": self.source_split, "units": {}}


def build_fleet_simulator(data_dir: str, subset: str, n_units: int = 6):
    """Factory: use real CMAPSS data for the given subset if the files
    exist, otherwise fall back to synthetic generation."""
    import os
    test_path = os.path.join(data_dir, f"test_{subset}.txt")
    train_path = os.path.join(data_dir, f"train_{subset}.txt")
    if os.path.exists(test_path) or os.path.exists(train_path):
        return RealDataFleetSimulator(data_dir=data_dir, subset=subset, n_units=n_units)
    print(f"WARNING: no CMAPSS data files found for subset '{subset}' in {data_dir}; "
          f"falling back to synthetic telemetry.")
    return SyntheticFleetSimulator(n_units=n_units)

