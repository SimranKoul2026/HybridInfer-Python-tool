"""Conformance tests: the Python implementation must pass every shared vector.

The same `conformance/vectors.json` is used by the Kotlin (AAR) test suite, so
passing here guarantees behavioral parity of the deterministic core. See SPEC.md.
"""
from __future__ import annotations

import json
import os
import unittest

from hybridinfer.complexity import ComplexityThresholds, complexity_bin, prompt_tokens
from hybridinfer.controller import ControllerConfig, FailureAwareController
from hybridinfer.reliability.risk import RiskProfile
from hybridinfer.reliability.state import SafetyStateMachine

VECTORS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "conformance", "vectors.json")


def _load():
    with open(VECTORS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _text(case) -> str:
    if "repeat" in case:
        return case["repeat"] * case["times"]
    return case["text"]


class _DummyBackend:
    """Minimal backend for prefer_local vectors (never actually streamed)."""

    def __init__(self, tier, name, model):
        self.tier, self.name, self.model = tier, name, model

    def stream(self, messages, *, timeout_s, stall_timeout_s=None, params=None):  # pragma: no cover
        return iter(())


class ConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = _load()

    def test_complexity(self):
        block = self.v["complexity"]
        th = ComplexityThresholds(block["short_max_tokens"], block["medium_max_tokens"])
        for c in block["cases"]:
            msgs = [{"role": "user", "content": _text(c)}]
            self.assertEqual(prompt_tokens(msgs), c["expected_tokens"], msg=str(c)[:80])
            self.assertEqual(complexity_bin(msgs, th), c["expected_bin"], msg=str(c)[:80])

    def test_risk(self):
        block = self.v["risk"]
        tol = block["tolerance"]
        for c in block["cases"]:
            rp = RiskProfile()
            for _ in range(c["failures"]):
                rp.update("ollama", "m", c["bin"], failed=True)
            for _ in range(c["attempts"] - c["failures"]):
                rp.update("ollama", "m", c["bin"], failed=False)
            got = rp.pr_fail("ollama", "m", c["bin"])
            self.assertLess(abs(got - c["expected_pfail"]), tol, msg="%s -> %r" % (c, got))

    def test_state(self):
        block = self.v["state"]
        p = block["params"]
        for case in block["cases"]:
            clock = {"t": 0.0}
            sm = SafetyStateMachine(
                caution_pfail=p["caution_pfail"],
                unsafe_failures=p["unsafe_failures"],
                recovery_cooldown_s=p["recovery_cooldown_s"],
                recovery_backoff=p["recovery_backoff"],
                recovery_cooldown_max_s=p["recovery_cooldown_max_s"],
                clock=lambda: clock["t"],
            )
            for step in case["steps"]:
                if "now" in step:
                    clock["t"] = float(step["now"])
                op = step["op"]
                if op == "on_decision":
                    sm.on_decision(step["pfail"])
                elif op == "on_local_failure":
                    sm.on_local_failure()
                elif op == "on_local_success":
                    sm.on_local_success()
                elif op == "local_allowed":
                    self.assertEqual(sm.local_allowed(), step["expect"],
                                     msg="%s @ %s" % (case["name"], step))
                else:
                    self.fail("unknown op %s" % op)
                self.assertEqual(sm.state.value, step["expect_state"],
                                 msg="%s after %s" % (case["name"], step))

    def test_prefer_local(self):
        for c in self.v["prefer_local"]["cases"]:
            cfg_fields = c["config"]
            cfg = ControllerConfig(
                force_local=cfg_fields.get("force_local", False),
                force_remote=cfg_fields.get("force_remote", False),
                enable_runtime_health_gating=cfg_fields.get("enable_runtime_health_gating", True),
                risk_prefer_remote=cfg_fields.get("risk_prefer_remote", 0.6),
            )
            risk = RiskProfile()
            rspec = c["risk"]
            for _ in range(rspec["failures"]):
                risk.update("ollama", "m", rspec["bin"], failed=True)
            for _ in range(rspec["attempts"] - rspec["failures"]):
                risk.update("ollama", "m", rspec["bin"], failed=False)
            ctrl = FailureAwareController(
                _DummyBackend("local", "ollama", "m"),
                _DummyBackend("remote", "openai", "r"),
                config=cfg,
                risk=risk,
            )
            self.assertEqual(
                ctrl._prefer_local(c["bin"]), c["expected_prefer_local"], msg=str(c)[:120]
            )


if __name__ == "__main__":
    unittest.main()
