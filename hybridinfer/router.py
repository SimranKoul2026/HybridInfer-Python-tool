"""HybridRouter: build backends + controller from Settings and route requests."""
from __future__ import annotations

import os
from typing import Iterator, List, Optional

from .backends import build_backend
from .backends.base import GenerationResult, Message
from .complexity import ComplexityThresholds
from .config import Settings
from .controller import ControllerConfig, FailureAwareController, StreamChunk
from .reliability.risk import RiskProfile


class HybridRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.local = build_backend(settings.local, "local")
        self.remote = build_backend(settings.remote, "remote")

        risk_path = settings.risk_profile_path
        if risk_path:
            risk_path = os.path.expanduser(risk_path)
        self.controller = FailureAwareController(
            self.local,
            self.remote,
            config=ControllerConfig(
                local_timeout_s=settings.local_timeout_s,
                local_stall_timeout_s=settings.local_stall_timeout_s,
                remote_timeout_s=settings.remote_timeout_s,
                risk_prefer_remote=settings.risk_prefer_remote,
                enable_runtime_health_gating=settings.enable_runtime_health_gating,
                enable_in_request_fallback=settings.enable_in_request_fallback,
                enable_recovery=settings.enable_recovery,
                force_local=settings.force_local,
                force_remote=settings.force_remote,
            ),
            risk=RiskProfile(risk_path),
            thresholds=ComplexityThresholds(
                short_max_tokens=settings.short_max_tokens,
                medium_max_tokens=settings.medium_max_tokens,
            ),
        )

    def complete(self, messages: List[Message]) -> GenerationResult:
        return self.controller.complete(messages)

    def stream(self, messages: List[Message]) -> Iterator[StreamChunk]:
        return self.controller.stream(messages)

    def models(self) -> List[str]:
        out = []
        if self.local is not None:
            out.append("local:" + self.local.model)
        if self.remote is not None:
            out.append("remote:" + self.remote.model)
        return out

    def risk_snapshot(self):
        return self.controller.risk.snapshot()

    def state(self) -> str:
        return self.controller.state.state.value
