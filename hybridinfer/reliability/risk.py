"""Self-calibrating per-(backend, model, length-bin) failure-risk profile.

Blends a conservative prior with observed outcomes (Beta-style), so a router
dropped onto a new machine/model learns its own local failure limits. Persisted
to a small JSON file so it survives restarts. Ported from the research
RiskProfile (Kotlin).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, Optional, Tuple

# Conservative priors seeded from measurement: long prompts carry high wedge risk
# even on a cool device. [short, medium, long].
_PRIOR_FAIL = (0.02, 0.10, 0.55)
_PRIOR_WEIGHT = 8.0  # pseudo-count strength of the prior


class RiskProfile:
    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        # key = "backend|model|bin" -> (attempts, failures)
        self._stats: Dict[str, Tuple[int, int]] = {}
        self._load()

    @staticmethod
    def _key(backend: str, model: str, bin_: int) -> str:
        return "{}|{}|{}".format(backend, model, bin_)

    def pr_fail(self, backend: str, model: str, bin_: int) -> float:
        """Pr(local failure | backend, model, length-bin)."""
        bin_ = max(0, min(2, bin_))
        attempts, failures = self._stats.get(self._key(backend, model, bin_), (0, 0))
        prior = _PRIOR_FAIL[bin_]
        num = prior * _PRIOR_WEIGHT + failures
        den = _PRIOR_WEIGHT + attempts
        v = num / den if den else prior
        return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)

    def update(self, backend: str, model: str, bin_: int, failed: bool) -> None:
        """Record an observed local outcome (failed = timeout/stall/crash/error)."""
        bin_ = max(0, min(2, bin_))
        with self._lock:
            attempts, failures = self._stats.get(self._key(backend, model, bin_), (0, 0))
            attempts = min(attempts + 1, 10_000)
            if failed:
                failures = min(failures + 1, 10_000)
            self._stats[self._key(backend, model, bin_)] = (attempts, failures)
            self._persist_locked()

    def snapshot(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._stats)

    # --- persistence ---
    def _load(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for k, v in raw.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    self._stats[k] = (int(v[0]), int(v[1]))
        except Exception:
            # A corrupt profile should never crash the router; start fresh.
            pass

    def _persist_locked(self) -> None:
        if not self._path:
            return
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({k: list(v) for k, v in self._stats.items()}, fh)
            os.replace(tmp, self._path)
        except Exception:
            pass
