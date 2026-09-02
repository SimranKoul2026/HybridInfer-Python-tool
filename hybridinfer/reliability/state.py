"""Safety state machine with hysteresis and probe-based recovery.

LOCAL_ELIGIBLE -> CAUTION -> UNSAFE -> RECOVERING -> RESTORED

Once local inference has failed enough, the router stops sending traffic to it
(UNSAFE) and serves from remote. After a cooldown it allows a single *probe*
request (RECOVERING); if that succeeds it is RESTORED, otherwise it drops back
to UNSAFE and the cooldown grows (exponential backoff, capped). Ported from the
research SafetyStateMachine.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


class SafetyState(enum.Enum):
    LOCAL_ELIGIBLE = "LOCAL_ELIGIBLE"
    CAUTION = "CAUTION"
    UNSAFE = "UNSAFE"
    RECOVERING = "RECOVERING"
    RESTORED = "RESTORED"


@dataclass
class SafetyStateMachine:
    caution_pfail: float = 0.5           # predicted risk that trips soft CAUTION
    unsafe_failures: int = 2             # consecutive local failures -> UNSAFE
    recovery_cooldown_s: float = 60.0    # base UNSAFE dwell before a probe is allowed
    recovery_backoff: float = 2.0        # cooldown multiplier per consecutive failed probe
    recovery_cooldown_max_s: float = 600.0  # cap on the backed-off cooldown
    clock: Callable[[], float] = time.monotonic

    state: SafetyState = field(init=False, default=SafetyState.LOCAL_ELIGIBLE)
    _consec_fail: int = field(init=False, default=0)
    _unsafe_since: Optional[float] = field(init=False, default=None)
    _probe_failures: int = field(init=False, default=0)

    def _cooldown(self) -> float:
        """Current probe cooldown: base * backoff^(failed probes), capped."""
        return min(
            self.recovery_cooldown_s * (self.recovery_backoff ** self._probe_failures),
            self.recovery_cooldown_max_s,
        )

    def on_decision(self, pfail: float) -> None:
        """Soft, pre-request gating: elevated predicted risk -> CAUTION."""
        if self.state in (SafetyState.LOCAL_ELIGIBLE, SafetyState.RESTORED):
            if pfail >= self.caution_pfail:
                self.state = SafetyState.CAUTION

    def local_allowed(self) -> bool:
        """Whether a local attempt is permitted right now."""
        if self.state in (
            SafetyState.LOCAL_ELIGIBLE,
            SafetyState.CAUTION,
            SafetyState.RESTORED,
        ):
            return True
        # UNSAFE / RECOVERING: allow exactly one probe once the (backed-off) cooldown elapses.
        if self._unsafe_since is not None:
            if (self.clock() - self._unsafe_since) >= self._cooldown():
                self.state = SafetyState.RECOVERING
                return True
        return False

    def on_local_success(self) -> None:
        self._consec_fail = 0
        if self.state == SafetyState.RECOVERING:
            self.state = SafetyState.RESTORED
            self._unsafe_since = None
            self._probe_failures = 0        # recovered: reset the backoff
        elif self.state in (SafetyState.CAUTION, SafetyState.RESTORED):
            self.state = SafetyState.LOCAL_ELIGIBLE

    def on_local_failure(self) -> None:
        self._consec_fail += 1
        if self.state == SafetyState.RECOVERING:
            # Probe failed: back to UNSAFE, grow the cooldown.
            self._probe_failures += 1
            self.state = SafetyState.UNSAFE
            self._unsafe_since = self.clock()
        elif self._consec_fail >= self.unsafe_failures:
            # First entry into UNSAFE for this episode: start at the base cooldown.
            self._probe_failures = 0
            self.state = SafetyState.UNSAFE
            self._unsafe_since = self.clock()
        else:
            self.state = SafetyState.CAUTION
