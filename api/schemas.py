from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    unit_id: str = Field(..., description="Equipment/unit identifier, e.g. 'turbine-07'")
    cycle: int = Field(..., ge=0, description="Operating cycle / timestep counter")
    op_setting_1: float = 0.0
    op_setting_2: float = 0.0
    op_setting_3: float = 100.0
    sensor_1: float = 500.0
    sensor_2: float = 500.0
    sensor_3: float = 500.0
    sensor_4: float = 500.0
    sensor_5: float = 500.0
    sensor_6: float = 500.0
    sensor_7: float = 500.0
    sensor_8: float = 500.0
    sensor_9: float = 500.0
    sensor_10: float = 500.0
    sensor_11: float = 500.0
    sensor_12: float = 500.0
    sensor_13: float = 500.0
    sensor_14: float = 500.0
    sensor_15: float = 500.0
    sensor_16: float = 500.0
    sensor_17: float = 500.0
    sensor_18: float = 500.0
    sensor_19: float = 500.0
    sensor_20: float = 500.0
    sensor_21: float = 500.0


class PredictionResponse(BaseModel):
    unit_id: str
    cycle: int
    failure_probability: float
    failure_predicted: bool
    anomaly_score: float
    anomaly_detected: bool
    anomaly_severity: float
    risk_score: float
    status: str
    decision_threshold: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    mlflow_run_id: Optional[str] = None
    warning_window: Optional[int] = None
    dataset_subset: Optional[str] = None
    dataset_source_split: Optional[str] = None


class FleetSummary(BaseModel):
    total_units: int
    healthy: int
    warning: int
    critical: int
    avg_risk_score: float
