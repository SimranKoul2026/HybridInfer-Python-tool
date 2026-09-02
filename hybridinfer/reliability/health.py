"""Rolling runtime-health signals (failures, timeouts, latency, TTFT).

Cheap counters updated from each request outcome; used alongside the risk
profile to gauge whether local inference is currently healthy. Ported from the
research RuntimeHealthMonitor.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional


@dataclass
class HealthSnapshot:
    recent_failures: int
    recent_timeouts: int
    last_latency_ms: Optional[float]
    last_ttft_ms: Optional[float]
    consecutive_local: int


class RuntimeHealthMonitor:
    def __init__(
        self,
        window_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window = window_s
        self._clock = clock
        self._failures: Deque[float] = deque()
        self._timeouts: Deque[float] = deque()
        self._last_latency: Optional[float] = None
        self._last_ttft: Optional[float] = None
        self._consec_local = 0

    def _prune(self) -> None:
        cutoff = self._clock() - self._window
        for q in (self._failures, self._timeouts):
            while q and q[0] < cutoff:
                q.popleft()

    def record_result(
        self,
        ok: bool,
        error: Optional[str],
        latency_ms: float,
        ttft_ms: Optional[float],
    ) -> None:
        now = self._clock()
        self._last_latency = latency_ms
        if ttft_ms is not None:
            self._last_ttft = ttft_ms
        if not ok:
            self._failures.append(now)
            if error in ("timeout", "prefill_timeout", "stall"):
                self._timeouts.append(now)
        self._prune()

    def on_local_started(self) -> None:
        self._consec_local += 1

    def on_offloaded(self) -> None:
        self._consec_local = 0

    def snapshot(self) -> HealthSnapshot:
        self._prune()
        return HealthSnapshot(
            recent_failures=len(self._failures),
            recent_timeouts=len(self._timeouts),
            last_latency_ms=self._last_latency,
            last_ttft_ms=self._last_ttft,
            consecutive_local=self._consec_local,
        )
