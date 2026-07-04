"""In-memory routing statistics.

Records every /moderate decision so the escalation rate — the share of
traffic actually sent to the expensive Tier-2 adjudicator — is measured
rather than assumed. Counters are per-process and reset on restart; for a
multi-replica deployment export to Prometheus/OpenTelemetry instead (kept
in-memory here to avoid infrastructure this project doesn't need).
"""
from __future__ import annotations

import threading
from collections import Counter


class RouteStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._routes: Counter[str] = Counter()
        self._triggers: Counter[str] = Counter()

    def record(self, route: str, reasons: list[str]) -> None:
        with self._lock:
            self._routes[route] += 1
            for reason in reasons:
                # "low_confidence:0.550<0.7" -> "low_confidence"
                self._triggers[reason.split(":", 1)[0]] += 1

    def snapshot(self) -> dict:
        with self._lock:
            total = sum(self._routes.values())
            tier2 = self._routes["escalated"] + self._routes["audit"]
            return {
                "total": total,
                "routes": dict(self._routes),
                "tier2_rate": round(tier2 / total, 4) if total else 0.0,
                "trigger_counts": dict(self._triggers),
            }
