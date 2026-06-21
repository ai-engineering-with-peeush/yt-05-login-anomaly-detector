"""
features.py — behavioral feature extraction for the login anomaly detector.

Three functions are built live on camera (geo_velocity_kmh, device_novelty,
hour_deviation).  burst_rate is pre-built: it needs global event context that
per-user history alone can't provide, which makes it a good contrast point.

Feature vector fed to IsolationForest:
  [geo_velocity_kmh, device_novelty, hour_deviation, burst_rate]
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List

from src.models import LoginEvent, UserHistory


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points (km)."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Pre-built feature — captures collective / aggregate anomaly
# ---------------------------------------------------------------------------

def burst_rate(
    event: LoginEvent,
    all_events: List[LoginEvent],
    window_minutes: int = 10,
) -> float:
    """Number of distinct user accounts with login activity in the last window_minutes.

    Each individual login during a credential-stuffing burst looks completely
    ordinary on its own.  Five hundred of them across different accounts in ten
    minutes is an attack — but only visible in the aggregate count.  No
    single-event rule can ever catch this; you need this global sliding-window
    feature in the vector.

    Returns 0.0 when the window is empty (first events of the day).
    """
    window_start = event.ts - timedelta(minutes=window_minutes)
    active_users = {
        e.user_id
        for e in all_events
        if window_start <= e.ts < event.ts
    }
    return float(len(active_users))


# ---------------------------------------------------------------------------
# Live-coded features — shells only; built on camera
# ---------------------------------------------------------------------------

def geo_velocity_kmh(prev_login: LoginEvent, curr_login: LoginEvent) -> float:
    """Speed (km/h) required to travel between two consecutive login locations.

    Catches impossible travel: a login from Frankfurt followed 40 minutes later
    by a login from Singapore implies a ~12 000 km/h commute.

    Uses haversine() for great-circle distance.  Clamps the time denominator
    to 1e-6 hours to avoid division-by-zero on same-second logins.

    Returns 0.0 when prev_login is None (first login — no prior location).
    """
    distance_km = haversine(prev_login.lat, prev_login.lon,
                            curr_login.lat, curr_login.lon)
    hours = (curr_login.ts - prev_login.ts).total_seconds() / 3600
    return distance_km / max(hours, 1e-6)


def device_novelty(user_history: UserHistory, curr_login: LoginEvent) -> float:
    """1.0 if this device has never been seen for this user, 0.0 otherwise.

    Uses devices_before(ts) so history is strictly prior to curr_login —
    no future leakage.

    A 2 AM login on a new device scores 1.0 here *and* high on hour_deviation,
    which pushes the combined feature vector into anomaly territory for the
    Isolation Forest.
    """
    seen = user_history.devices_before(curr_login.ts)
    return 0.0 if curr_login.device_id in seen else 1.0


def hour_deviation(user_history: UserHistory, curr_login: LoginEvent) -> float:
    """Circular distance (hours) from the user's median login hour.

    Returns a value in [0, 12]: 0 means the login is at the user's typical
    hour; 12 means it's as far as possible (opposite side of the clock).

    A 2 AM login is normal for an on-call engineer — their median_login_hour
    might already be 2.  For an accountant who has worked 9-to-5 for three
    years, it deviates by 10+ hours.  Same timestamp, different context.
    That per-user framing is exactly what makes this a *contextual* anomaly,
    and also why a global model partially misses it for irregular workers
    (episode 2 fix: per-user baselines).
    """
    median_hour = user_history.median_login_hour(curr_login.ts)
    diff = abs(curr_login.ts.hour + curr_login.ts.minute / 60.0 - median_hour)
    return min(diff, 24 - diff)
