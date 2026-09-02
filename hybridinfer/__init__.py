"""HybridInfer: a reliability-aware LLM router.

Runs each request on your local model when it safely can, and automatically
falls back to a remote model the moment local inference stalls, crashes, or is
predicted to fail - so you get local-first efficiency without the crashes.

The routing + reliability core here is a Python port of the failure-aware
runtime-health controller from the HybridInfer research system.
"""
from __future__ import annotations

__version__ = "0.2.0"

# Core (stdlib-only) surface. Heavier pieces (backends/router/server) pull in
# httpx/fastapi and are imported from their own modules on demand.
from .backends.base import BackendError, GenerationResult  # noqa: E402
from .complexity import ComplexityThresholds, complexity_bin, prompt_tokens  # noqa: E402
from .controller import ControllerConfig, FailureAwareController, StreamChunk  # noqa: E402
from .reliability.risk import RiskProfile  # noqa: E402
from .reliability.state import SafetyState, SafetyStateMachine  # noqa: E402

__all__ = [
    "__version__",
    "BackendError",
    "GenerationResult",
    "StreamChunk",
    "ComplexityThresholds",
    "complexity_bin",
    "prompt_tokens",
    "ControllerConfig",
    "FailureAwareController",
    "RiskProfile",
    "SafetyState",
    "SafetyStateMachine",
]
