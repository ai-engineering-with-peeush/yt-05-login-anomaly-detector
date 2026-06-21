"""
features.py — STARTER FILE (live coding starting point)

burst_rate is pre-built.  Build the three functions below on camera.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import List

from src.models import LoginEvent, UserHistory


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points (km)."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def burst_rate(
    event: LoginEvent,
    all_events: List[LoginEvent],
    window_minutes: int = 10,
) -> float:
    """Number of distinct user accounts with login activity in the last window_minutes."""
    window_start = event.ts - timedelta(minutes=window_minutes)
    active_users = {
        e.user_id
        for e in all_events
        if window_start <= e.ts < event.ts
    }
    return float(len(active_users))


def geo_velocity_kmh(prev_login: LoginEvent, curr_login: LoginEvent) -> float:
    """Speed (km/h) required to travel between two consecutive login locations.

    Catches impossible travel: a login from Frankfurt followed 40 minutes later
    by a login from Singapore implies a ~12 000 km/h commute.

    Uses haversine() for great-circle distance.  Clamps the time denominator
    to 1e-6 hours to avoid division-by-zero on same-second logins.

    Returns 0.0 when prev_login is None (first login — no prior location).
    """
    pass


def device_novelty(user_history: UserHistory, curr_login: LoginEvent) -> float:
    """1.0 if this device has never been seen for this user, 0.0 otherwise.

    Uses devices_before(ts) so history is strictly prior to curr_login —
    no future leakage.
    """
    pass


def hour_deviation(user_history: UserHistory, curr_login: LoginEvent) -> float:
    """Circular distance (hours) from the user's median login hour.

    Returns a value in [0, 12]: 0 means the login is at the user's typical
    hour; 12 means it's as far as possible (opposite side of the clock).
    """
    pass
