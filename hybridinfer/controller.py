"""Failure-aware runtime-health controller.

Decides local vs remote per request, runs the local attempt under a
timeout + stall watchdog, and on any local failure transparently falls back to
remote (in-request). Records every outcome to self-calibrate the risk profile
and drive the safety state machine (which can pull local out of rotation and
probe it back in). Python port of the research FailureAwareController.

The feature flags mirror the research A0-A3 ablation arms:
  * force_local + everything off            ~ A0 (no control)
  * (no host-thermal signal on desktop)     ~ A1 (thermal-only, not applicable)
  * enable_runtime_health_gating + fallback ~ A2 (runtime-only)
  * all enabled                             ~ A3 (full)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

from .backends.base import Backend, BackendError, GenerationResult, Message
from .complexity import ComplexityThresholds, complexity_bin
from .reliability.health import RuntimeHealthMonitor
from .reliability.risk import RiskProfile
from .reliability.state import SafetyStateMachine


@dataclass
class StreamChunk:
    """One event from FailureAwareController.stream()."""

    delta: str = ""                       # a text chunk (empty on the final event)
    tier: str = ""                        # tier that produced this delta
    model: str = ""                       # model that produced this delta
    done: bool = False                    # True only on the final event
    error: Optional[str] = None           # set if the committed tier failed
    meta: Optional[Dict[str, object]] = None  # routing metadata on the final event


@dataclass
class ControllerConfig:
    local_timeout_s: float = 120.0
    local_stall_timeout_s: float = 20.0   # inter-token gap that counts as a wedge
    remote_timeout_s: float = 120.0
    risk_prefer_remote: float = 0.6       # pFail at/above which we skip local up front
    enable_runtime_health_gating: bool = True
    enable_in_request_fallback: bool = True
    enable_recovery: bool = True
    force_local: bool = False
    force_remote: bool = False


class FailureAwareController:
    def __init__(
        self,
        local: Optional[Backend],
        remote: Optional[Backend],
        *,
        config: Optional[ControllerConfig] = None,
        risk: Optional[RiskProfile] = None,
        health: Optional[RuntimeHealthMonitor] = None,
        state: Optional[SafetyStateMachine] = None,
        thresholds: Optional[ComplexityThresholds] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.local = local
        self.remote = remote
        self.cfg = config or ControllerConfig()
        self.risk = risk or RiskProfile()
        self.health = health or RuntimeHealthMonitor(clock=clock)
        self.state = state or SafetyStateMachine(clock=clock)
        self.thresholds = thresholds or ComplexityThresholds()
        self._clock = clock

    def complete(self, messages: List[Message]) -> GenerationResult:
        bin_ = complexity_bin(messages, self.thresholds)
        route: List[str] = []

        if self._prefer_local(bin_) and self.local is not None and self._local_permitted():
            self.health.on_local_started()
            res = self.local.generate(
                messages,
                timeout_s=self.cfg.local_timeout_s,
                stall_timeout_s=self.cfg.local_stall_timeout_s,
            )
            route.append("local")
            self.health.record_result(res.ok, res.error, res.latency_ms, res.ttft_ms)
            self.risk.update(self.local.name, self.local.model, bin_, failed=not res.ok)

            if res.ok:
                self.state.on_local_success()
                res.route = route
                res.tier = "local"
                return res

            # local failed
            self.state.on_local_failure()
            self.health.on_offloaded()
            if not self.cfg.enable_in_request_fallback or self.remote is None:
                res.route = route
                return res  # no fallback configured: surface the failure honestly

        # remote tier (either preferred, or the fallback after a local failure)
        if self.remote is not None:
            res = self.remote.generate(messages, timeout_s=self.cfg.remote_timeout_s)
            route.append("remote")
            self.health.record_result(res.ok, res.error, res.latency_ms, res.ttft_ms)
            res.route = route
            res.tier = "remote"
            res.fell_back = "local" in route
            return res

        # only a local tier exists and we hadn't tried it yet (e.g. gated off): try it
        if self.local is not None and "local" not in route:
            self.health.on_local_started()
            res = self.local.generate(
                messages,
                timeout_s=self.cfg.local_timeout_s,
                stall_timeout_s=self.cfg.local_stall_timeout_s,
            )
            route.append("local")
            self.risk.update(self.local.name, self.local.model, bin_, failed=not res.ok)
            self.health.record_result(res.ok, res.error, res.latency_ms, res.ttft_ms)
            if res.ok:
                self.state.on_local_success()
            else:
                self.state.on_local_failure()
            res.route = route
            res.tier = "local"
            return res

        return GenerationResult(ok=False, error="no_backend_available", route=route)

    def stream(self, messages: List[Message]) -> Iterator[StreamChunk]:
        """Streaming variant with first-token-commit fallback semantics.

        If local fails BEFORE emitting a token (the dominant prefill-wedge case),
        we fall back to remote cleanly - the client has seen nothing yet. Once a
        local token has been emitted the request is committed to local; a
        mid-stream wedge ends the stream honestly (it cannot be un-sent).
        """
        bin_ = complexity_bin(messages, self.thresholds)
        route: List[str] = []
        tried_local = False

        if self._prefer_local(bin_) and self.local is not None and self._local_permitted():
            tried_local = True
            route.append("local")
            self.health.on_local_started()
            emitted = 0
            pre_token_error: Optional[str] = None
            mid_error: Optional[str] = None
            try:
                for delta in self.local.stream(
                    messages,
                    timeout_s=self.cfg.local_timeout_s,
                    stall_timeout_s=self.cfg.local_stall_timeout_s,
                ):
                    emitted += 1
                    yield StreamChunk(delta=delta, tier="local", model=self.local.model)
            except BackendError as e:
                if emitted > 0:
                    mid_error = e.code          # committed; cannot fall back
                else:
                    pre_token_error = e.code    # clean fallback still possible

            if emitted > 0:
                ok = mid_error is None
                self.risk.update(self.local.name, self.local.model, bin_, failed=not ok)
                self.health.record_result(ok, mid_error, 0.0, None)
                if ok:
                    self.state.on_local_success()
                else:
                    self.state.on_local_failure()
                yield StreamChunk(
                    done=True,
                    error=mid_error,
                    meta=self._meta("local", route, False, emitted),
                )
                return

            # zero tokens from local: a pre-token error, or an empty completion
            code = pre_token_error or "empty"
            self.risk.update(self.local.name, self.local.model, bin_, failed=True)
            self.health.record_result(False, code, 0.0, None)
            self.state.on_local_failure()
            self.health.on_offloaded()
            if not self.cfg.enable_in_request_fallback or self.remote is None:
                yield StreamChunk(done=True, error=code, meta=self._meta("local", route, False, 0))
                return
            # else fall through to remote

        if self.remote is not None:
            route.append("remote")
            n = 0
            err: Optional[str] = None
            try:
                for delta in self.remote.stream(messages, timeout_s=self.cfg.remote_timeout_s):
                    n += 1
                    yield StreamChunk(delta=delta, tier="remote", model=self.remote.model)
            except BackendError as e:
                err = e.code
            self.health.record_result(err is None, err, 0.0, None)
            yield StreamChunk(
                done=True,
                error=err,
                meta=self._meta("remote", route, tried_local, n),
            )
            return

        # local-only existed but was gated off, and there is no remote: last-resort local
        if self.local is not None and not tried_local:
            route.append("local")
            n = 0
            err = None
            try:
                for delta in self.local.stream(
                    messages,
                    timeout_s=self.cfg.local_timeout_s,
                    stall_timeout_s=self.cfg.local_stall_timeout_s,
                ):
                    n += 1
                    yield StreamChunk(delta=delta, tier="local", model=self.local.model)
            except BackendError as e:
                err = e.code
            ok = err is None and n > 0
            self.risk.update(self.local.name, self.local.model, bin_, failed=not ok)
            if ok:
                self.state.on_local_success()
            else:
                self.state.on_local_failure()
            yield StreamChunk(
                done=True,
                error=err if err else (None if ok else "empty"),
                meta=self._meta("local", route, False, n),
            )
            return

        yield StreamChunk(done=True, error="no_backend_available", meta=self._meta("", route, False, 0))

    def _meta(self, tier: str, route: List[str], fell_back: bool, n: int) -> Dict[str, object]:
        return {
            "tier": tier,
            "route": list(route),
            "fell_back": fell_back,
            "completion_tokens": n,
            "state": self.state.state.value,
        }

    # --- routing policy ---
    def _prefer_local(self, bin_: int) -> bool:
        if self.cfg.force_remote:
            return False
        if self.cfg.force_local:
            return True
        if self.local is None:
            return False
        if self.remote is None:
            return True
        if self.cfg.enable_runtime_health_gating:
            pfail = self.risk.pr_fail(self.local.name, self.local.model, bin_)
            self.state.on_decision(pfail)
            if pfail >= self.cfg.risk_prefer_remote:
                return False
        return True

    def _local_permitted(self) -> bool:
        """Recovery gate: when enabled, an UNSAFE local tier is held out until a
        cooldown probe. When disabled, local is always permitted if preferred."""
        if not self.cfg.enable_recovery:
            return True
        return self.state.local_allowed()
