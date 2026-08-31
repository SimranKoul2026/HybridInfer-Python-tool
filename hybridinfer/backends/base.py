"""Backend interface + the uniform result the controller reasons about.

`stream()` is the primitive every backend implements: it yields text deltas and
raises BackendError on failure. `generate()` (non-streaming) is derived from it,
so both paths share identical error/timing handling.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from ..complexity import prompt_tokens

Message = Dict[str, Any]

# error codes the controller treats as a local failure worth falling back on
FAILURE_ERRORS = {"timeout", "stall", "connection", "oom", "server_error", "empty"}


class BackendError(Exception):
    """Raised by a backend's stream() on failure. `code` is one of FAILURE_ERRORS
    (or an http_* code); `detail` is optional human context."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


@dataclass
class GenerationResult:
    text: str = ""
    ok: bool = False
    error: Optional[str] = None        # None on success; else an error code
    tier: str = ""                     # "local" | "remote"
    backend: str = ""                  # "ollama" | "openai" | ...
    model: str = ""
    ttft_ms: Optional[float] = None    # time to first token
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    fell_back: bool = False            # True if a local attempt failed first
    route: List[str] = field(default_factory=list)  # tiers tried, in order


class Backend(ABC):
    """A single inference tier. Implementations enforce their own timeouts and,
    for streaming local backends, stall (token-rate-collapse) detection."""

    tier: str = ""
    name: str = ""    # backend kind, e.g. "ollama"
    model: str = ""

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        *,
        timeout_s: float,
        stall_timeout_s: Optional[float] = None,
    ) -> Iterator[str]:
        """Yield text deltas as they arrive. Raise BackendError on failure."""
        ...

    def available(self) -> bool:
        """Cheap reachability check (best-effort)."""
        return True

    def generate(
        self,
        messages: List[Message],
        *,
        timeout_s: float,
        stall_timeout_s: Optional[float] = None,
        on_token: Optional[Callable[[str, int], None]] = None,
    ) -> GenerationResult:
        """Non-streaming convenience: consume stream() into one result."""
        start = time.monotonic()
        ttft: Optional[float] = None
        chunks: List[str] = []
        n = 0
        try:
            for delta in self.stream(
                messages, timeout_s=timeout_s, stall_timeout_s=stall_timeout_s
            ):
                now = time.monotonic()
                if ttft is None:
                    ttft = (now - start) * 1000.0
                n += 1
                chunks.append(delta)
                if on_token is not None:
                    on_token(delta, n)
        except BackendError as e:
            return GenerationResult(
                text="".join(chunks),
                ok=False,
                error=e.code,
                tier=self.tier,
                backend=self.name,
                model=self.model,
                ttft_ms=ttft,
                latency_ms=(time.monotonic() - start) * 1000.0,
                prompt_tokens=prompt_tokens(messages),
                completion_tokens=n,
            )
        return GenerationResult(
            text="".join(chunks),
            ok=True,
            error=None,
            tier=self.tier,
            backend=self.name,
            model=self.model,
            ttft_ms=ttft,
            latency_ms=(time.monotonic() - start) * 1000.0,
            prompt_tokens=prompt_tokens(messages),
            completion_tokens=n,
        )
