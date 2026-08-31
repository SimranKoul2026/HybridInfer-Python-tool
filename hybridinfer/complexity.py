"""Prompt-complexity estimation.

A cheap, dependency-free proxy for how expensive a request is to serve locally.
Longer inputs dominate prefill cost and (per the research) are where on-device
runtimes wedge, so complexity drives whether we try local first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

Message = Dict[str, Any]

BIN_NAMES = ["short", "medium", "long"]


def estimate_tokens(text: str) -> int:
    """Rough token count without a tokenizer (~4 chars/token, floored by words).

    Uses explicit round-half-up then floor (int() truncates toward zero for
    positive values) so the result is identical to the Kotlin port. See SPEC.md.
    """
    if not text:
        return 0
    chars = len(text)
    words = len(text.split())
    return max(1, int(max(chars / 4.0, words * 0.75) + 0.5))


def messages_text(messages: List[Message]) -> str:
    """Flatten OpenAI-style messages (str or content-part list) to plain text."""
    parts: List[str] = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for seg in content:
                if isinstance(seg, dict) and seg.get("type") == "text":
                    parts.append(str(seg.get("text", "")))
    return "\n".join(parts)


def prompt_tokens(messages: List[Message]) -> int:
    return estimate_tokens(messages_text(messages))


@dataclass
class ComplexityThresholds:
    """Token boundaries for the short/medium/long complexity bins."""

    short_max_tokens: int = 128
    medium_max_tokens: int = 512


def complexity_bin(
    messages: List[Message], thresholds: Optional[ComplexityThresholds] = None
) -> int:
    """0 = short, 1 = medium, 2 = long."""
    thresholds = thresholds or ComplexityThresholds()
    t = prompt_tokens(messages)
    if t < thresholds.short_max_tokens:
        return 0
    if t < thresholds.medium_max_tokens:
        return 1
    return 2
