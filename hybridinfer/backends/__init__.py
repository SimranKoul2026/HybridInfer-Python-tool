"""Inference backends (tiers). Import concrete backends lazily to avoid pulling
in httpx unless a networked backend is actually constructed."""
from __future__ import annotations

from typing import Optional

from .base import Backend, GenerationResult, Message


def build_backend(spec, tier: str) -> Optional[Backend]:
    """Construct a backend from a config spec. Returns None if spec is falsy."""
    if spec is None:
        return None
    kind = (spec.backend or "").lower()
    if kind == "ollama":
        from .ollama import OllamaBackend

        return OllamaBackend(
            model=spec.model,
            base_url=spec.base_url or "http://127.0.0.1:11434",
            tier=tier,
        )
    if kind in ("openai", "openai_compat", "openai-compatible"):
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(
            model=spec.model,
            base_url=spec.base_url or "https://api.openai.com/v1",
            api_key=spec.resolved_api_key(),
            tier=tier,
        )
    raise ValueError("unknown backend kind: {!r}".format(spec.backend))


__all__ = ["Backend", "GenerationResult", "Message", "build_backend"]
