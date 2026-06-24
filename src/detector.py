"""
detector.py — Isolation Forest login anomaly detector.

Functions built here mirror what run_eval.py uses, but this module owns
the canonical training path and exposes score_event() for serve.py.

What gets built
---------------
1.  load_events()         — read data/logs.csv -> List[LoginEvent]
2.  build_histories()     — group events by user -> Dict[str, UserHistory]
3.  extract_feature_row() — single event -> [geo_vel, device_nov, hour_dev, burst]
4.  build_feature_matrix()— training events -> (X_train, event_ids)
5.  IsolationForest fit   — n_estimators=200, contamination=0.002
6.  z-score baseline      — threshold on geo_velocity alone
7.  score_event()         — one event -> decision dict + structured JSON log
8.  train_and_save()      — orchestrates 1-7, pickles model to data/model.pkl
"""

import bisect
import csv
import json
import logging
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

# sys.path.insert lets `python src/detector.py` resolve `src.*` imports.
# Kept at module level because the from-src imports below need it at parse time.
sys.path.insert(0, str(Path(__file__).parent.parent))

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
DATA_DIR      = Path(__file__).parent.parent / "data"
MODEL_PATH    = DATA_DIR / "model.pkl"
MODEL_VERSION = "v1"
TS_FMT        = "%Y-%m-%dT%H:%M:%S"

N_ESTIMATORS  = 200
CONTAMINATION = 0.002
RANDOM_STATE  = 42
TRAIN_DAYS    = 20
START_DATE    = datetime(2026, 5, 1)


# ---------------------------------------------------------------------------
# 1. load_events
# ---------------------------------------------------------------------------

def load_events(path: Path = DATA_DIR / "logs.csv") -> List[LoginEvent]:
    """Read logs.csv and return events in source order."""
    events: List[LoginEvent] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            events.append(LoginEvent(
                event_id  = row["event_id"],
                user_id   = row["user_id"],
                ts        = datetime.strptime(row["ts"], TS_FMT),
                device_id = row["device_id"],
                country   = row["country"],
                lat       = float(row["lat"]),
                lon       = float(row["lon"]),
                success   = row["success"].lower() == "true",
            ))
    return events


# ---------------------------------------------------------------------------
# 2. build_histories
# ---------------------------------------------------------------------------

def build_histories(events: List[LoginEvent]) -> Dict[str, UserHistory]:
    """Group events by user_id and wrap each group in a UserHistory."""
    by_user: Dict[str, List[LoginEvent]] = defaultdict(list)
    for e in events:
        by_user[e.user_id].append(e)
    return {uid: UserHistory(evts) for uid, evts in by_user.items()}


# ---------------------------------------------------------------------------
# 3. extract_feature_row
# ---------------------------------------------------------------------------

def extract_feature_row(
    event: LoginEvent,
    window_events: List[LoginEvent],
    histories: Dict[str, UserHistory],
) -> Optional[List[float]]:
    """Single event -> [geo_velocity_kmh, device_novelty, hour_deviation, burst_rate].

    window_events: the burst-rate candidate pool — pass the full event list for
                   online scoring, or a pre-bisected 10-minute slice for batch
                   scoring to keep build_feature_matrix at O(N log N).

    Returns None when the event has no prior history (first login per user).
    Those rows are skipped during training: fitting on a zero-signal stub
    would artificially inflate the "normal" density around the origin.
    """
    history = histories.get(event.user_id)
    if history is None:
        return None

    prior = history.events_before(event.ts)
    if not prior:
        return None

    prev = prior[-1]   # most recent prior login

    return [
        geo_velocity_kmh(prev, event),
        device_novelty(history, event),
        hour_deviation(history, event),
        burst_rate(event, window_events),
    ]


# ---------------------------------------------------------------------------
# 4. build_feature_matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    events: List[LoginEvent],
    histories: Dict[str, UserHistory],
) -> Tuple[np.ndarray, List[str]]:
    """Training-window events -> (X_train, event_ids).

    X_train  : feature rows for events in days 0-19.
               Attack events are absent by design — generate_logs injects
               them only into days 20-29.
    event_ids: event_id strings aligned with X_train rows.

    burst_rate is O(N log N) overall: events are sorted once, then each
    event's 10-minute window is located in O(log N) via bisect.
    """
    train_cutoff = START_DATE + timedelta(days=TRAIN_DAYS)

    # Sort all events once and build a timestamp index for bisect.
    sorted_events = sorted(events, key=lambda e: e.ts)
    ts_list = [e.ts for e in sorted_events]

    X_train, eids = [], []
    for event in sorted_events:
        if event.ts >= train_cutoff:
            break  # events are sorted; nothing past the cutoff is needed

        # Locate the 10-minute burst-rate window in O(log N).
        w_start = event.ts - timedelta(minutes=10)
        lo = bisect.bisect_left(ts_list, w_start)
        hi = bisect.bisect_left(ts_list, event.ts)

        row = extract_feature_row(event, sorted_events[lo:hi], histories)
        if row is None:
            continue
        X_train.append(row)
        eids.append(event.event_id)

    return np.array(X_train), eids


# ---------------------------------------------------------------------------
# 7. score_event
# ---------------------------------------------------------------------------

def score_event(
    event: LoginEvent,
    all_events: List[LoginEvent],
    histories: Dict[str, UserHistory],
    detector: IsolationForest,
    threshold: float,
) -> dict:
    """Score one event and emit a structured JSON log line.

    The log line is the audit trail: when an alert fires at 2 AM, this is how
    you know which feature drove it without re-running the pipeline.
    """
    row = extract_feature_row(event, all_events, histories)

    if row is None:
        decision = {
            "event_id":      event.id,
            "user_id":       event.user_id,
            "score":         None,
            "alert":         False,
            "features":      None,
            "model_version": MODEL_VERSION,
        }
        logger.info(json.dumps(decision))
        return decision

    raw_score = float(detector.score_samples([row])[0])
    is_alert  = raw_score < threshold

    decision = {
        "event_id": event.id,
        "user_id":  event.user_id,
        "score":    round(raw_score, 4),
        "alert":    is_alert,
        "features": {
            "geo_velocity_kmh": round(row[0], 2),
            "device_novelty":   round(row[1], 2),
            "hour_deviation":   round(row[2], 2),
            "burst_rate":       round(row[3], 2),
        },
        "model_version": MODEL_VERSION,
    }
    logger.info(json.dumps(decision))
    return decision


# ---------------------------------------------------------------------------
# 8. train_and_save
# ---------------------------------------------------------------------------

def train_and_save() -> None:
    """Orchestrate data loading, feature extraction, training, and model persistence."""
    print("Loading events...")
    events = load_events()
    print(f"  {len(events):,} events loaded")

    print("Building user histories...")
    histories = build_histories(events)

    print("Extracting training feature matrix...")
    X_train, _ = build_feature_matrix(events, histories)
    print(f"  {len(X_train):,} training rows")

    # ------------------------------------------------------------------
    # 5. IsolationForest fit
    # ------------------------------------------------------------------
    print(f"Fitting IsolationForest (n_estimators={N_ESTIMATORS}, contamination={CONTAMINATION})...")
    detector = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        contamination = CONTAMINATION,
        random_state  = RANDOM_STATE,
    )
    detector.fit(X_train)

    # Threshold: score at the contamination percentile of training scores.
    # Any event scoring below this is flagged as anomalous.
    train_scores = detector.score_samples(X_train)
    threshold    = float(np.percentile(train_scores, CONTAMINATION * 100))
    print(f"  Alert threshold: {threshold:.4f}")

    # ------------------------------------------------------------------
    # 6. Z-score baseline stats on geo_velocity alone (for reference)
    # ------------------------------------------------------------------
    geo_vels = X_train[:, 0]
    z_mu     = float(np.mean(geo_vels))
    z_std    = float(np.std(geo_vels))
    print(f"  Geo-velocity baseline: mu={z_mu:.1f} km/h  std={z_std:.1f} km/h")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    MODEL_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "detector":  detector,
        "threshold": threshold,
        "z_mu":      z_mu,
        "z_std":     z_std,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
