"""
run_eval.py — honest evaluation of the login anomaly detector.

Run via:  make eval   (or: python evals/run_eval.py)

What this harness does
----------------------
1.  Loads data/logs.csv and data/labels.csv
2.  Trains IsolationForest on NORMAL events only (days 0-19, no attacks)
3.  Replays ALL events through the trained detector
4.  Compares predictions against ground-truth labels
5.  Prints per-attack-type precision/recall, alerts-per-10k, and a
    comparison against a simple z-score baseline on geo_velocity alone

The labels file is opened only here — the detector never sees it during
training or scoring.  "Evaluation against known cases, not vibes."
"""

from __future__ import annotations

import csv
import json
import logging
import math
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

# ── project imports ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LoginEvent, UserHistory
from src.features import geo_velocity_kmh, device_novelty, hour_deviation, burst_rate

logging.basicConfig(level=logging.WARNING)   # suppress INFO spam during eval

DATA_DIR   = Path(__file__).parent.parent / "data"
TS_FMT     = "%Y-%m-%dT%H:%M:%S"
TRAIN_DAYS = 20   # first 20 days = normal-only training window
START_DATE = datetime(2026, 5, 1)

# IsolationForest config — must match what's built live in detector.py
N_ESTIMATORS  = 200
CONTAMINATION = 0.002
RANDOM_STATE  = 42


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_events(path: Path) -> List[LoginEvent]:
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


def load_labels(path: Path) -> Dict[str, str]:
    """Returns {event_id: attack_type}."""
    labels: Dict[str, str] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            labels[row["event_id"]] = row["attack_type"]
    return labels


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_histories(events: List[LoginEvent]) -> Dict[str, UserHistory]:
    by_user: Dict[str, List[LoginEvent]] = defaultdict(list)
    for e in events:
        by_user[e.user_id].append(e)
    return {uid: UserHistory(evts) for uid, evts in by_user.items()}


def extract_row(
    event: LoginEvent,
    all_events: List[LoginEvent],
    histories: Dict[str, UserHistory],
) -> Optional[List[float]]:
    """Extract a 4-dimensional feature vector for one event.

    Returns None if the event has no prior history (first login per user) —
    these are excluded from training and flagged as 0-score in eval.
    """
    history = histories.get(event.user_id)
    if history is None:
        return None

    prior = history.events_before(event.ts)
    if not prior:
        return None

    prev = prior[-1]   # most recent prior login

    # --- features built live on camera ---
    vel   = geo_velocity_kmh(prev, event)
    nov   = device_novelty(history, event)
    hdev  = hour_deviation(history, event)
    # --- pre-built aggregate feature ---
    burst = burst_rate(event, all_events)

    return [vel, nov, hdev, burst]


# ---------------------------------------------------------------------------
# Train + score
# ---------------------------------------------------------------------------

def train(
    events: List[LoginEvent],
    labels: Dict[str, str],
    all_events: List[LoginEvent],
    histories: Dict[str, UserHistory],
) -> Tuple[IsolationForest, float, np.ndarray, List[str]]:
    """Train on normal events only (training window, no attack labels)."""
    train_cutoff = START_DATE + timedelta(days=TRAIN_DAYS)

    X_train, X_all, eids = [], [], []
    for event in events:
        row = extract_row(event, all_events, histories)
        if row is None:
            continue
        X_all.append(row)
        eids.append(event.event_id)
        # Only include normal events in the training window
        if event.ts < train_cutoff and event.event_id not in labels:
            X_train.append(row)

    X_train_arr = np.array(X_train)
    X_all_arr   = np.array(X_all)

    detector = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    detector.fit(X_train_arr)

    # Alert threshold: score at the contamination percentile of training scores
    train_scores = detector.score_samples(X_train_arr)
    threshold = float(np.percentile(train_scores, CONTAMINATION * 100))

    return detector, threshold, X_all_arr, eids


# ---------------------------------------------------------------------------
# Z-score baseline
# ---------------------------------------------------------------------------

def baseline_alerts(
    events: List[LoginEvent],
    labels: Dict[str, str],
    all_events: List[LoginEvent],
    histories: Dict[str, UserHistory],
    z_threshold: float = 3.0,
) -> Dict[str, bool]:
    """Simple baseline: z-score on geo_velocity_kmh alone (no other features)."""
    vels, eids = [], []
    for event in events:
        history = histories.get(event.user_id)
        if not history:
            continue
        prior = history.events_before(event.ts)
        if not prior:
            continue
        vel = geo_velocity_kmh(prior[-1], event)
        vels.append(vel)
        eids.append(event.event_id)

    arr = np.array(vels)
    z   = np.abs(stats.zscore(arr))
    return {eid: bool(z[i] > z_threshold) for i, eid in enumerate(eids)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def print_report(
    all_events: List[LoginEvent],
    labels: Dict[str, str],
    scored_eids: List[str],
    if_alerts: Dict[str, bool],
    baseline: Dict[str, bool],
) -> None:
    attack_types = ["impossible_travel", "new_device_odd_hour", "credential_stuffing"]

    print("\n" + "=" * 65)
    print("  Login Anomaly Detector — Evaluation Report")
    print("=" * 65)

    # ── Per-attack metrics ───────────────────────────────────────────────
    print(f"\n{'Attack':<30} {'P':>6} {'R':>6} {'F1':>6}   {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 65)

    total_alerts = sum(1 for a in if_alerts.values() if a)

    for attack in attack_types:
        attack_eids = {eid for eid, t in labels.items() if t == attack}
        tp = sum(1 for eid in scored_eids if eid in attack_eids and if_alerts.get(eid, False))
        fp = sum(1 for eid in scored_eids if eid not in labels and if_alerts.get(eid, False))
        fn = sum(1 for eid in attack_eids if not if_alerts.get(eid, False))
        p, r, f = _prf(tp, fp, fn)
        print(f"  {attack:<28} {p:>5.0%} {r:>6.0%} {f:>6.0%}   {tp:>4} {fp:>4} {fn:>4}")

    # ── Operational metric ───────────────────────────────────────────────
    total_scored = len(scored_eids)
    alerts_per_10k = total_alerts / total_scored * 10_000
    print(f"\n  Total events scored : {total_scored:,}")
    print(f"  Total alerts        : {total_alerts:,}")
    print(f"  Alerts per 10k      : {alerts_per_10k:.1f}")
    print(f"  (contamination={CONTAMINATION} → budget {CONTAMINATION*10_000:.0f}/10k)")

    # ── Baseline comparison ──────────────────────────────────────────────
    print(f"\n{'Baseline (geo_velocity z-score)':<30} {'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 50)
    for attack in attack_types:
        attack_eids = {eid for eid, t in labels.items() if t == attack}
        b_scored = set(baseline.keys())
        tp = sum(1 for eid in b_scored if eid in attack_eids and baseline.get(eid, False))
        fp = sum(1 for eid in b_scored if eid not in labels and baseline.get(eid, False))
        fn = sum(1 for eid in attack_eids if not baseline.get(eid, False))
        p, r, f = _prf(tp, fp, fn)
        print(f"  {attack:<28} {p:>5.0%} {r:>6.0%} {f:>6.0%}")

    print("\n" + "=" * 65)
    print("  Missed new_device_odd_hour cases = irregular-schedule users.")
    print("  Their median_login_hour already spans odd hours → low deviation.")
    print("  Fix: per-user behavioral baselines (Episode 2).")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    events = load_events(DATA_DIR / "logs.csv")
    labels = load_labels(DATA_DIR / "labels.csv")
    print(f"  {len(events):,} events  |  {len(labels):,} labeled attacks")

    print("Building user histories...")
    histories = build_histories(events)

    print("Training IsolationForest on normal events (days 0-19)...")
    detector, threshold, X_all, scored_eids = train(events, labels, events, histories)
    print(f"  Alert threshold: {threshold:.4f}")

    print("Scoring all events...")
    raw_scores = detector.score_samples(X_all)
    if_alerts = {eid: bool(raw_scores[i] < threshold) for i, eid in enumerate(scored_eids)}

    print("Running z-score baseline...")
    baseline = baseline_alerts(events, labels, events, histories)

    print_report(events, labels, scored_eids, if_alerts, baseline)

    # Persist model for serve.py
    model_payload = {"detector": detector, "threshold": threshold}
    with open(DATA_DIR / "model.pkl", "wb") as f:
        pickle.dump(model_payload, f)
    print(f"Model saved → {DATA_DIR / 'model.pkl'}")
