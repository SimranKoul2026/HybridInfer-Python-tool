"""Local tier via Ollama (streaming).

Streams the response so we can measure time-to-first-token and, crucially,
detect a *token-rate collapse* - a wedge where the runtime stops emitting tokens
without finishing. httpx's per-read timeout gives us that for free: if no bytes
arrive for `stall_timeout_s`, we raise BackendError("stall") and let the
controller fall back.
"""
from __future__ import annotations

import json
import time
from typing import Iterator, List, Optional

from .base import Backend, BackendError, Message


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        tier: str = "local",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.tier = tier

    def available(self) -> bool:
        try:
            import httpx

            r = httpx.get(self.base_url + "/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def stream(
        self,
        messages: List[Message],
        *,
        timeout_s: float,
        stall_timeout_s: Optional[float] = None,
    ) -> Iterator[str]:
        import httpx

        url = self.base_url + "/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": True}
        read_to = stall_timeout_s or timeout_s
        timeout = httpx.Timeout(timeout_s, connect=min(10.0, timeout_s), read=read_to)
        start = time.monotonic()

        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, json=payload) as r:
                    if r.status_code != 200:
                        body = r.read().decode("utf-8", "ignore")
                        code = "oom" if "memory" in body.lower() else "server_error"
                        raise BackendError(code, body[:200])
                    for line in r.iter_lines():
                        if time.monotonic() - start > timeout_s:
                            raise BackendError("timeout")
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except ValueError:
                            continue
                        if obj.get("error"):
                            err = str(obj["error"])
                            code = "oom" if "memory" in err.lower() else "server_error"
                            raise BackendError(code, err[:200])
                        chunk = (obj.get("message") or {}).get("content", "")
                        if chunk:
                            yield chunk
                        if obj.get("done"):
                            return
        except httpx.ReadTimeout:
            # No token for `read_to` seconds: token-rate collapse / wedge.
            raise BackendError("stall")
        except httpx.TimeoutException:
            raise BackendError("timeout")
        except httpx.HTTPError:
            raise BackendError("connection")
