"""
models.py — shared data structures for yt-05-login-anomaly-detector.

LoginEvent   — one row from data/logs.csv
UserHistory  — per-user event index with strictly-before lookups
               (avoids future leakage, the same trap that haunts every ML pipeline)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Set


@dataclass
class LoginEvent:
    event_id: str
    user_id: str
    ts: datetime
    device_id: str
    country: str
    lat: float
    lon: float
    success: bool

    @property
    def id(self) -> str:
        """Alias used by score_event for cleaner log output."""
        return self.event_id


class UserHistory:
    """Per-user event index for feature extraction.

    All lookups are *strictly before* the given timestamp to prevent leaking
    future data into features — the same trap that shows up in every ML system
    and is invisible until production numbers mysteriously underperform tests.
    """

    def __init__(self, events: List[LoginEvent]) -> None:
        self._events: List[LoginEvent] = sorted(events, key=lambda e: e.ts)

    def events_before(self, ts: datetime) -> List[LoginEvent]:
        """All events strictly before ts."""
        return [e for e in self._events if e.ts < ts]

    def devices_before(self, ts: datetime) -> Set[str]:
        """Set of device IDs seen strictly before ts."""
        return {e.device_id for e in self.events_before(ts)}

    def median_login_hour(self, ts: datetime) -> float:
        """Median hour-of-day (0–23) across logins before ts.

        Returns 12.0 (noon) when there is no prior history — a neutral default
        that neither penalises night-shift workers nor ignores new accounts.
        """
        prior = self.events_before(ts)
        if not prior:
            return 12.0
        hours = sorted(e.ts.hour + e.ts.minute / 60.0 for e in prior)
        mid = len(hours) // 2
        if len(hours) % 2 == 0:
            return (hours[mid - 1] + hours[mid]) / 2.0
        return hours[mid]
