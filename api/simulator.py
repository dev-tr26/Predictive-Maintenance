# simulates live telemetry for fleet of equipment units, each progressing through a degradation curve like the offline training data.
# Stands in for a real telemetry source (PLC/SCADA/OPC-UA/MQTT) in this demo to show in real time indashboard.
from __future__ import annotations
import numpy as np

from src.preprocessing import SENSOR_COLS, SETTING_COLS

# Sensors that trend during degradation, matching data/generate_synthetic_cmapss.py
DEGRADING_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 15, 17, 20, 21]
EQUIPMENT_TYPES = ["turbine", "compressor", "pump", "generator"]


class FleetSimulator:
    def __init__(self, n_units: int = 6, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.unit_ids = [f"unit-{i+1:02d}" for i in range(n_units)]
        self.equipment_type = {
            uid: EQUIPMENT_TYPES[i % len(EQUIPMENT_TYPES)] for i, uid in enumerate(self.unit_ids)
        }
        self.cycle: dict[str, int] = {uid: 0 for uid in self.unit_ids}
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
        """Advance one unit by one cycle and return (reading_dict, cycle)."""
        self.cycle[uid] += 1
        cycle = self.cycle[uid]
        life = self.max_life[uid]

        if cycle >= life:
            # failure reached -> maintenance performed -> unit resets
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
            if sensor_idx in DEGRADING_SENSORS:
                trend = self.trend_dir[uid][i] * self.trend_mag[uid][i] * degradation
            reading[col] = float(level + trend + noise)

        for i, col in enumerate(SETTING_COLS):
            base = [0.0, 0.0, 100.0][i]
            scale = [0.002, 0.0003, 0.02][i]
            reading[col] = float(self.rng.normal(base, scale))

        return reading, cycle
