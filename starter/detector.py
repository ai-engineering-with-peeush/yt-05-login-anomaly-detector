"""
detector.py — STARTER FILE (live coding starting point)

Imports, logger, and constants are pre-typed.  Build everything below on camera.
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR      = Path(__file__).parent.parent / "data"
MODEL_VERSION = "v1"
ALERT_THRESHOLD: float = 0.0

# Build live below ↓
