"""
detector.py — Isolation Forest login anomaly detector.

*** STARTER FILE — built live on camera ***

Everything below the dashed line is written during the recording.
The imports, logger, and constants below are pre-typed so the demo
starts from something that already runs cleanly.

What gets built live
--------------------
1.  load_events()         — read data/logs.csv → List[LoginEvent]
2.  build_histories()     — group events by user → Dict[str, UserHistory]
3.  extract_feature_row() — single event → [geo_vel, device_nov, hour_dev, burst]
4.  build_feature_matrix()— all events → np.ndarray  (train/test split)
5.  IsolationForest fit   — n_estimators=200, contamination=0.002
6.  z-score baseline      — threshold on geo_velocity alone
7.  score_event()         — one event → decision dict + structured JSON log
8.  train_and_save()      — orchestrates 1-7, pickles model to data/model.pkl
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

from src.models import LoginEvent, UserHistory
from src.features import geo_velocity_kmh, device_novelty, hour_deviation, burst_rate

# ---------------------------------------------------------------------------
# Logger — every scored event emits one JSON line.
# When a 2 AM page asks "why did this alert fire?" the answer is already in
# the log: score + the individual feature values that produced it.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent.parent / "data"
MODEL_PATH  = DATA_DIR / "model.pkl"
MODEL_VERSION = "v1"
TS_FMT      = "%Y-%m-%dT%H:%M:%S"

# ALERT_THRESHOLD is set after training (score_samples threshold at contamination)
ALERT_THRESHOLD: float = 0.0   # overwritten by train_and_save()

# ---------------------------------------------------------------------------
# Build live on camera ↓
# ---------------------------------------------------------------------------
