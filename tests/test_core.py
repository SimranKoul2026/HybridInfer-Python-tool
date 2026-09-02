"""Core-logic tests. Stdlib + package only - no network, no httpx/fastapi.

Run with:  python -m pytest    (or)   python -m unittest discover -s tests
"""
from __future__ import annotations

import os
import tempfile
import unittest
from typing import List, Optional

from hybridinfer.backends.base import Backend, BackendError, GenerationResult, Message
from hybridinfer.complexity import complexity_bin, prompt_tokens
from hybridinfer.controller import ControllerConfig, FailureAwareController
from hybridinfer.reliability.risk import RiskProfile
from hybridinfer.reliability.state import SafetyState, SafetyStateMachine


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class ScriptedBackend(Backend):
    """A backend whose per-call behaviour is scripted via `stream()`.

    Each outcome is one of:
      True            -> yields a couple of tokens then completes
      "fail"          -> fails before any token (clean-fallback case)
      "midfail"       -> yields one token then fails (committed case)
    """

    def __init__(self, tier: str, name: str, model: str, outcomes: List[object]) -> None:
        self.tier = tier
        self.name = name
        self.model = model
        self._outcomes = list(outcomes)
        self.calls = 0

    def stream(self, messages, *, timeout_s, stall_timeout_s=None):
        outcome = self._outcomes[self.calls] if self.calls < len(self._outcomes) else True
        self.calls += 1
        if outcome == "fail":
            raise BackendError("stall")
        if outcome == "midfail":
            yield "part"
            raise BackendError("stall")
        # success
        yield "ok"
        yield "!"


class ComplexityTests(unittest.TestCase):
    def test_bins(self) -> None:
        short = [{"role": "user", "content": "hi"}]
        long_text = "word " * 4000
        self.assertEqual(complexity_bin(short), 0)
        self.assertEqual(complexity_bin([{"role": "user", "content": long_text}]), 2)
        self.assertGreater(prompt_tokens([{"role": "user", "content": long_text}]), 512)


class RiskProfileTests(unittest.TestCase):
    def test_prior_and_update(self) -> None:
        rp = RiskProfile()
        # long-bin prior is high
        self.assertAlmostEqual(rp.pr_fail("ollama", "m", 2), 0.55, places=2)
        # observing successes pulls risk down
        for _ in range(20):
            rp.update("ollama", "m", 2, failed=False)
        self.assertLess(rp.pr_fail("ollama", "m", 2), 0.2)

    def test_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "risk.json")
            rp = RiskProfile(path)
            rp.update("ollama", "m", 1, failed=True)
            self.assertTrue(os.path.exists(path))
            rp2 = RiskProfile(path)  # reload
            self.assertEqual(rp2.snapshot(), rp.snapshot())


class StateMachineTests(unittest.TestCase):
    def test_unsafe_then_probe_recovery(self) -> None:
        clk = FakeClock()
        sm = SafetyStateMachine(unsafe_failures=2, recovery_cooldown_s=60.0, clock=clk)
        sm.on_local_failure()
        self.assertEqual(sm.state, SafetyState.CAUTION)
        sm.on_local_failure()
        self.assertEqual(sm.state, SafetyState.UNSAFE)
        self.assertFalse(sm.local_allowed())      # held out during cooldown
        clk.advance(61.0)
        self.assertTrue(sm.local_allowed())        # probe permitted
        self.assertEqual(sm.state, SafetyState.RECOVERING)
        sm.on_local_success()                      # probe succeeds
        self.assertEqual(sm.state, SafetyState.RESTORED)


class ControllerTests(unittest.TestCase):
    def test_local_success_stays_local(self) -> None:
        local = ScriptedBackend("local", "ollama", "m", [True])
        remote = ScriptedBackend("remote", "openai", "r", [True])
        ctrl = FailureAwareController(local, remote, config=ControllerConfig())
        res = ctrl.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(res.ok)
        self.assertEqual(res.tier, "local")
        self.assertFalse(res.fell_back)
        self.assertEqual(remote.calls, 0)

    def test_local_failure_falls_back_to_remote(self) -> None:
        local = ScriptedBackend("local", "ollama", "m", ["fail"])
        remote = ScriptedBackend("remote", "openai", "r", [True])
        ctrl = FailureAwareController(local, remote, config=ControllerConfig())
        res = ctrl.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(res.ok)
        self.assertEqual(res.tier, "remote")
        self.assertTrue(res.fell_back)
        self.assertEqual(res.route, ["local", "remote"])

    def test_no_fallback_surfaces_failure(self) -> None:
        local = ScriptedBackend("local", "ollama", "m", ["fail"])
        remote = ScriptedBackend("remote", "openai", "r", [True])
        cfg = ControllerConfig(enable_in_request_fallback=False)
        ctrl = FailureAwareController(local, remote, config=cfg)
        res = ctrl.complete([{"role": "user", "content": "hi"}])
        self.assertFalse(res.ok)
        self.assertEqual(res.route, ["local"])
        self.assertEqual(remote.calls, 0)

    def test_high_risk_prefers_remote_upfront(self) -> None:
        # Force the long-bin prior (0.55) below threshold so a low threshold routes remote first.
        local = ScriptedBackend("local", "ollama", "m", [True])
        remote = ScriptedBackend("remote", "openai", "r", [True])
        cfg = ControllerConfig(risk_prefer_remote=0.5)  # long prior 0.55 >= 0.5 -> remote
        ctrl = FailureAwareController(local, remote, config=cfg)
        res = ctrl.complete([{"role": "user", "content": "word " * 4000}])
        self.assertEqual(res.tier, "remote")
        self.assertEqual(local.calls, 0)


class StreamingTests(unittest.TestCase):
    @staticmethod
    def _drain(ctrl, messages):
        deltas, final = [], None
        for ch in ctrl.stream(messages):
            if ch.done:
                final = ch
            else:
                deltas.append(ch.delta)
        return deltas, final

    def test_stream_local_success(self) -> None:
        local = ScriptedBackend("local", "ollama", "m", [True])
        remote = ScriptedBackend("remote", "openai", "r", [True])
        ctrl = FailureAwareController(local, remote, config=ControllerConfig())
        deltas, final = self._drain(ctrl, [{"role": "user", "content": "hi"}])
        self.assertEqual(deltas, ["ok", "!"])
        self.assertEqual(final.meta["tier"], "local")
        self.assertFalse(final.meta["fell_back"])
        self.assertIsNone(final.error)
        self.assertEqual(remote.calls, 0)

    def test_stream_pre_token_failure_falls_back_clean(self) -> None:
        # local fails BEFORE any token -> client should see ONLY remote's tokens
        local = ScriptedBackend("local", "ollama", "m", ["fail"])
        remote = ScriptedBackend("remote", "openai", "r", [True])
        ctrl = FailureAwareController(local, remote, config=ControllerConfig())
        deltas, final = self._drain(ctrl, [{"role": "user", "content": "hi"}])
        self.assertEqual(deltas, ["ok", "!"])            # no local tokens leaked
        self.assertEqual(final.meta["tier"], "remote")
        self.assertTrue(final.meta["fell_back"])
        self.assertEqual(final.meta["route"], ["local", "remote"])

    def test_stream_midstream_failure_commits_no_fallback(self) -> None:
        # local emits a token then wedges -> committed to local, no fallback
        local = ScriptedBackend("local", "ollama", "m", ["midfail"])
        remote = ScriptedBackend("remote", "openai", "r", [True])
        ctrl = FailureAwareController(local, remote, config=ControllerConfig())
        deltas, final = self._drain(ctrl, [{"role": "user", "content": "hi"}])
        self.assertEqual(deltas, ["part"])               # the partial token was sent
        self.assertEqual(final.meta["tier"], "local")
        self.assertFalse(final.meta["fell_back"])
        self.assertEqual(final.error, "stall")
        self.assertEqual(remote.calls, 0)                # cannot un-send; no fallback


class PrefillFailBackend(Backend):
    """A local engine whose first token never arrives (TTFT/prefill timeout)."""

    tier, name, model = "local", "ollama", "m"

    def stream(self, messages, *, timeout_s, stall_timeout_s=None):
        raise BackendError("prefill_timeout")
        yield ""  # unreachable; makes this a generator function


class V2Tests(unittest.TestCase):
    """v0.2: idempotency contract + structured routing reason + error classes."""

    def _ctrl(self, local_outcomes):
        local = ScriptedBackend("local", "ollama", "m", local_outcomes)
        remote = ScriptedBackend("remote", "openai", "r", [True])
        return FailureAwareController(local, remote, config=ControllerConfig()), local, remote

    def test_reason_local_ok(self):
        ctrl, _, remote = self._ctrl([True])
        res = ctrl.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(res.reason, "local_ok")
        self.assertEqual(remote.calls, 0)

    def test_reason_fell_back(self):
        ctrl, _, _ = self._ctrl(["fail"])
        res = ctrl.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(res.fell_back)
        self.assertTrue(res.reason.startswith("fell_back:"))

    def test_idempotency_no_fallback_on_side_effecting(self):
        # non-idempotent op + no key -> do NOT replay on remote; surface the failure
        ctrl, _, remote = self._ctrl(["fail"])
        res = ctrl.complete([{"role": "user", "content": "mutate"}], safe_to_retry=False)
        self.assertFalse(res.ok)
        self.assertEqual(res.route, ["local"])
        self.assertTrue(res.reason.startswith("no_fallback_unsafe:"))
        self.assertEqual(remote.calls, 0)

    def test_idempotency_key_reenables_fallback(self):
        ctrl, _, remote = self._ctrl(["fail"])
        res = ctrl.complete(
            [{"role": "user", "content": "hi"}], safe_to_retry=False, idempotency_key="k1"
        )
        self.assertTrue(res.ok)
        self.assertEqual(res.tier, "remote")
        self.assertTrue(res.fell_back)
        self.assertEqual(res.idempotency_key, "k1")
        self.assertEqual(remote.calls, 1)

    def test_prefill_timeout_counts_as_failure_and_falls_back(self):
        remote = ScriptedBackend("remote", "openai", "r", [True])
        ctrl = FailureAwareController(PrefillFailBackend(), remote, config=ControllerConfig())
        res = ctrl.complete([{"role": "user", "content": "hi"}])
        self.assertTrue(res.ok)
        self.assertEqual(res.tier, "remote")
        self.assertTrue(res.reason.startswith("fell_back:prefill_timeout"))

    def test_stream_idempotency_no_fallback(self):
        ctrl, _, remote = self._ctrl(["fail"])  # pre-token local failure
        deltas, final = [], None
        for ch in ctrl.stream([{"role": "user", "content": "mutate"}], safe_to_retry=False):
            if ch.done:
                final = ch
            else:
                deltas.append(ch.delta)
        self.assertEqual(deltas, [])
        self.assertTrue(final.meta["reason"].startswith("no_fallback_unsafe"))
        self.assertEqual(remote.calls, 0)


if __name__ == "__main__":
    unittest.main()
