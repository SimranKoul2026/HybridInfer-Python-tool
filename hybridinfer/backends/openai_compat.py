"""Remote tier via any OpenAI-compatible chat-completions endpoint (streaming).

Works with OpenAI, OpenRouter, Together, a local vLLM server, etc. - anything
that speaks POST {base_url}/chat/completions with `stream: true` (SSE).
"""
from __future__ import annotations

import json
from typing import Iterator, List, Optional

from .base import Backend, BackendError, Message


class OpenAICompatBackend(Backend):
    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        tier: str = "remote",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tier = tier

    def stream(
        self,
        messages: List[Message],
        *,
        timeout_s: float,
        stall_timeout_s: Optional[float] = None,
    ) -> Iterator[str]:
        import httpx

        url = self.base_url + "/chat/completions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        payload = {"model": self.model, "messages": messages, "stream": True}

        try:
            with httpx.Client(timeout=timeout_s) as client:
                with client.stream("POST", url, json=payload, headers=headers) as r:
                    if r.status_code != 200:
                        body = r.read().decode("utf-8", "ignore")
                        if r.status_code >= 500:
                            code = "server_error"
                        elif r.status_code in (401, 403):
                            code = "auth"
                        else:
                            code = "http_%d" % r.status_code
                        raise BackendError(code, body[:300])
                    for line in r.iter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            return
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except ValueError:
                            continue
                        try:
                            delta = obj["choices"][0]["delta"].get("content")
                        except (KeyError, IndexError, TypeError):
                            delta = None
                        if delta:
                            yield delta
        except httpx.TimeoutException:
            raise BackendError("timeout")
        except httpx.HTTPError:
            raise BackendError("connection")
