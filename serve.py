"""
serve.py — FastAPI scoring service for the login anomaly detector.

Run via:  make serve   (or: uvicorn serve:app --reload)

Requires model.pkl to exist — run 'make eval' first.

The endpoint accepts a login event and returns the same decision dict that
score_event produces, through the same structured logging path.  Serving
done *properly* — registry, CI, rollback — is its own episode later in the
series; this wires the end-to-end path today.

Example:
    curl -s -X POST http://localhost:8000/score \
      -H "Content-Type: application/json" \
      -d '{"event_id":"e1","user_id":"user_001","ts":"2026-05-01T10:30:00",
           "device_id":"dev-user_001-a","country":"Germany",
           "lat":52.52,"lon":13.40,"success":true}'
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models import LoginEvent, UserHistory
from src.features import device_novelty, hour_deviation, burst_rate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = DATA_DIR / "model.pkl"
MODEL_VERSION = "v1"
TS_FMT = "%Y-%m-%dT%H:%M:%S"

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
if not MODEL_PATH.exists():
    raise RuntimeError("model.pkl not found — run 'make eval' first.")

with open(MODEL_PATH, "rb") as _f:
    _payload = pickle.load(_f)

_detector = _payload["detector"]
_threshold = _payload["threshold"]

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    event_id: str
    user_id: str
    ts: str  # ISO 8601, e.g. "2026-05-01T10:30:00"
    device_id: str
    country: str
    lat: float
    lon: float
    success: bool = True


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Login Anomaly Detector", version=MODEL_VERSION)


@app.post("/score")
def score(req: LoginRequest) -> dict:
    """Score a single login event and return the anomaly decision."""
    try:
        ts = datetime.strptime(req.ts, TS_FMT)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"ts must be {TS_FMT}")

    event = LoginEvent(
        event_id=req.event_id,
        user_id=req.user_id,
        ts=ts,
        device_id=req.device_id,
        country=req.country,
        lat=req.lat,
        lon=req.lon,
        success=req.success,
    )

    # Minimal history: only the current event (no prior context in this call).
    # A production system would pass a real history store; this shows the path.
    history = UserHistory([event])
    features = [
        0.0,  # geo_velocity: no prior location
        device_novelty(history, event),
        hour_deviation(history, event),
        burst_rate(event, [event]),
    ]

    raw_score = float(_detector.score_samples([features])[0])
    is_alert = raw_score < _threshold

    decision = {
        "event_id": event.event_id,
        "user_id": event.user_id,
        "score": round(raw_score, 4),
        "alert": is_alert,
        "features": {
            "geo_velocity_kmh": round(features[0], 2),
            "device_novelty": round(features[1], 2),
            "hour_deviation": round(features[2], 2),
            "burst_rate": round(features[3], 2),
        },
        "model_version": MODEL_VERSION,
    }
    logger.info(json.dumps(decision))
    return decision


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}
