"""
FASTApi backend Endpoints:

  GET  /api/health              liveness + model-load status
  POST /api/predict             score a single sensor reading
  POST /api/predict/batch       score a list of readings
  GET  /api/fleet/summary       aggregate status across all simulated units
  GET  /api/fleet/units         current state of every unit
  GET  /api/fleet/history/{id}  recent time-series for one unit (for charts)
  GET  /api/metrics             latest offline model evaluation metrics
  WS   /ws/live                 real-time push feed of predictions
    
"""

from __future__ import annotations
import asyncio
import json
import os
import time
from collections import deque
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.simulator import build_fleet_simulator
from api.schemas import SensorReading, PredictionResponse, HealthResponse, FleetSummary
# from api.inference import engine

from api.inference import PredictiveMaintenanceEngine


MODEL_DIR = os.environ.get("MODEL_DIR", "./models_3")
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "./artifacts_3")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
TICK_SECONDS = float(os.environ.get("SIM_TICK_SECONDS", "1.5"))
HISTORY_LEN = 200

engine = PredictiveMaintenanceEngine(model_dir=MODEL_DIR)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global fleet_sim
    if os.path.exists(os.path.join(MODEL_DIR, "xgb_classifier.joblib")):
        engine.load()
        print(f"Models loaded from {MODEL_DIR}")
    else:
        print(f"WARNING: no trained models found in {MODEL_DIR}. "
              f"Run `python -m src.train` first. API will return 503 on /predict.")
    
    
    # Stream whichever subset the loaded model was actually trained on
    # (config.json's "subset" field, written by src/train.py --subset), so a
    # model trained on FD002 gets scored against real FD002 test data, etc.
    # Falls back to FD001 if config predates the --subset flag.
    subset = engine.config.get("subset", "FD001") if engine.loaded else "FD001"
    fleet_sim = build_fleet_simulator(data_dir=DATA_DIR, subset=subset, n_units=6)
    print(f"Fleet simulator streaming real '{subset}' data "
          f"({fleet_sim.info().get('source_split', 'n/a')} split)")

    
    for uid in fleet_sim.unit_ids:
        history[uid] = deque(maxlen=HISTORY_LEN)
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()
    
        
app = FastAPI(
    title="Industrial Equipment Failure Prediction API",
    description="Real-time predictive maintenance for turbines, pumps, compressors and generators.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

fleet_sim = None  # constructed in lifespan(), once we know which subset the model was trained on
history: Dict[str, deque] = {}
connected_clients: List[WebSocket] = []
latest_predictions: Dict[str, dict] = {}


# bg tasks : virtual fleet + scores each unit keeps rolling history for dashboard pushes updates to all connected websocket clients
# instead of kafka / rabbitmq
async def simulation_loop():
    while True:
        if engine.loaded:
            batch = []
            for uid in fleet_sim.unit_ids:
                reading, cycle = fleet_sim.step(uid)
                try:
                    pred = engine.score_reading(uid, reading, cycle)
                except Exception as e:
                    pred = {"unit_id": uid, "cycle": cycle, "error": str(e)}
                pred["timestamp"] = time.time()
                pred["equipment_type"] = fleet_sim.equipment_type[uid]
                latest_predictions[uid] = pred
                history[uid].append(pred)
                batch.append(pred)

            payload = json.dumps({"type": "tick", "predictions": batch})
            stale = []
            for ws in connected_clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                if ws in connected_clients:
                    connected_clients.remove(ws)
        await asyncio.sleep(TICK_SECONDS)
        
 
        
@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded=engine.loaded,
        mlflow_run_id=engine.config.get("mlflow_run_id") if engine.loaded else None,
        warning_window=engine.config.get("warning_window") if engine.loaded else None,
        dataset_subset=fleet_sim.subset if fleet_sim else None,
        dataset_source_split=fleet_sim.source_split if fleet_sim else None,
    )
    

@app.post("/api/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    if not engine.loaded:
        raise HTTPException(503, "Models not loaded. Train models first via `python -m src.train`.")
    payload = reading.model_dump()
    unit_id = payload.pop("unit_id")
    cycle = payload.pop("cycle")
    result = engine.score_reading(unit_id, payload, cycle)
    return PredictionResponse(**result)


@app.post("/api/predict/batch", response_model=List[PredictionResponse])
def predict_batch(readings: List[SensorReading]):
    if not engine.loaded:
        raise HTTPException(503, "Models not loaded. Train models first via `python -m src.train`.")
    results = []
    for reading in readings:
        payload = reading.model_dump()
        unit_id = payload.pop("unit_id")
        cycle = payload.pop("cycle")
        results.append(PredictionResponse(**engine.score_reading(unit_id, payload, cycle)))
    return results



@app.get("/api/fleet/summary", response_model=FleetSummary)
def fleet_summary():
    preds = list(latest_predictions.values())
    if not preds:
        return FleetSummary(total_units=len(fleet_sim.unit_ids),healthy=0,warning=0,critical=0,avg_risk_score=0.0)
    statuses = [p.get("status", "healthy") for p in preds]
    return FleetSummary(
        total_units=len(preds),
        healthy=statuses.count("healthy"),
        warning=statuses.count("warning"),
        critical=statuses.count("critical"),
        avg_risk_score=round(float(np.mean([p.get("risk_score", 0.0) for p in preds])), 4),
    )
    

@app.get("/api/fleet/units")
def fleet_units():
    return {"units": list(latest_predictions.values())}


@app.get("/api/fleet/info")
def fleet_units():
    if fleet_sim is None:
        raise HTTPException(503, "Fleet simulator not initialized.")
    return fleet_sim.info()


# Which CMAPSS subset/split is streaming, and which real unit_number each dashboard slot is replaying -- useful for confirming the model and the streamed data actually match.
@app.get("/api/fleet/history/{unit_id}")
def fleet_history(unit_id: str):
    if unit_id not in history:
        raise HTTPException(404, f"Unknown unit_id '{unit_id}'")
    return {"unit_id": unit_id, "history": list(history[unit_id])}


@app.get("/api/metrics")
def model_metrics():
    out = {}
    for name in ("classifier_metrics.json", "autoencoder_metrics.json"):
        path = os.path.join(ARTIFACTS_DIR, name)
        if os.path.exists(path):
            with open(path) as f:
                out[name.replace(".json", "")] = json.load(f)
    if not out:
        raise HTTPException(404, "No evaluation metrics found. Run training first.")
    return out

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    try:
        # send curr snapshot directly to dashboard not wait for next tick 
        await ws.send_text(json.dumps({"type": "snapshot", "predictions": list(latest_predictions.values())}))
        while True:
            await ws.receive_text() # keeping conn alive 
    except WebSocketDisconnect:
        if ws in connected_clients:
            connected_clients.remove(ws)
        
        
# serving dashboard static files
DASHBOARD_DIR = os.environ.get("DASHBOARD_DIR", "./dashboard")
if os.path.isdir(DASHBOARD_DIR):
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

    @app.get("/")
    def dashboard_index():
        return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))
