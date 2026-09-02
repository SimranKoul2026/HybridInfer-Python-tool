"""OpenAI-compatible HTTP server.

Exposes POST /v1/chat/completions so any OpenAI client (or app) can point at
HybridInfer and get transparent local-first routing with automatic fallback.
Also serves /v1/models, /healthz, and /stats.

Idempotency (v0.2): send `Idempotency-Key: <key>` to mark a request safe to
retry, or `X-HybridInfer-Safe-To-Retry: false` to mark a side-effecting request
that must NOT be replayed on fallback. (Both can also be set in the request body
under a `hybridinfer` object.) Every response carries the routing `reason`.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .config import Settings
from .router import HybridRouter


def _resolve_idempotency(
    key_hdr: Optional[str], safe_hdr: Optional[str], body: Dict[str, Any]
) -> Tuple[Optional[str], bool]:
    """(idempotency_key, safe_to_retry) from headers, falling back to a body block."""
    hi = body.get("hybridinfer") if isinstance(body.get("hybridinfer"), dict) else {}
    key = key_hdr or hi.get("idempotency_key")
    raw = safe_hdr if safe_hdr is not None else hi.get("safe_to_retry")
    if isinstance(raw, str):
        safe = raw.strip().lower() not in ("false", "0", "no")
    elif isinstance(raw, bool):
        safe = raw
    else:
        safe = True   # default: chat is read-only / safe to retry
    return key, safe


def _sse_chunk(cid: str, created: int, model: str, delta: Dict[str, Any], finish: Any, extra: Any = None) -> str:
    obj: Dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if extra is not None:
        obj["hybridinfer"] = extra
    return "data: " + json.dumps(obj) + "\n\n"


def _event_stream(
    router: HybridRouter, messages: List[Dict[str, Any]],
    idempotency_key: Optional[str], safe_to_retry: bool,
) -> Iterator[str]:
    """Format the controller's StreamChunks as OpenAI-style SSE."""
    created = int(time.time())
    cid = "hybridinfer-%d" % created
    model = "hybridinfer"
    role_sent = False
    meta: Dict[str, Any] = {}
    for ch in router.stream(messages, idempotency_key=idempotency_key, safe_to_retry=safe_to_retry):
        if ch.done:
            meta = ch.meta or {}
            if ch.error:
                meta = dict(meta)
                meta["error"] = ch.error
            break
        if not ch.delta:
            continue
        if ch.tier:
            model = "%s:%s" % (ch.tier, ch.model)
        if not role_sent:
            yield _sse_chunk(cid, created, model, {"role": "assistant"}, None)
            role_sent = True
        yield _sse_chunk(cid, created, model, {"content": ch.delta}, None)
    # final chunk carries finish_reason + hybridinfer routing metadata (incl. reason)
    yield _sse_chunk(cid, created, model, {}, "stop", extra=meta)
    yield "data: [DONE]\n\n"


def create_app(settings: Settings):
    from fastapi import FastAPI, Header
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="HybridInfer", version="0.2.0")
    router = HybridRouter(settings)

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"status": "ok", "state": router.state(), "models": router.models()}

    @app.get("/v1/models")
    def models() -> Dict[str, Any]:
        data = []
        for m in router.models():
            data.append({"id": m, "object": "model", "owned_by": "hybridinfer"})
        return {"object": "list", "data": data}

    @app.get("/stats")
    def stats() -> Dict[str, Any]:
        return {
            "state": router.state(),
            "risk_profile": {k: list(v) for k, v in router.risk_snapshot().items()},
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: Dict[str, Any],
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
        safe_to_retry_hdr: Optional[str] = Header(default=None, alias="X-HybridInfer-Safe-To-Retry"),
    ) -> Any:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "`messages` is required", "type": "invalid_request_error"}},
            )

        idem_key, safe_to_retry = _resolve_idempotency(idempotency_key, safe_to_retry_hdr, body)

        # Streaming: return SSE. Starlette iterates the sync generator in a
        # threadpool, so the blocking backend HTTP stays off the event loop.
        if body.get("stream"):
            return StreamingResponse(
                _event_stream(router, messages, idem_key, safe_to_retry),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # The router does blocking HTTP; run it off the event loop.
        import anyio

        res = await anyio.to_thread.run_sync(
            lambda: router.complete(messages, idempotency_key=idem_key, safe_to_retry=safe_to_retry)
        )

        if not res.ok:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "request not completed (last error: %s)" % res.error,
                        "type": "upstream_error",
                        "hybridinfer": {
                            "route": res.route,
                            "error": res.error,
                            "reason": res.reason,
                            "idempotency_key": res.idempotency_key,
                        },
                    }
                },
            )

        created = int(time.time())
        return {
            "id": "hybridinfer-%d" % created,
            "object": "chat.completion",
            "created": created,
            "model": "%s:%s" % (res.tier, res.model),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": res.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "total_tokens": res.prompt_tokens + res.completion_tokens,
            },
            # HybridInfer routing metadata (non-standard, safe for clients to ignore)
            "hybridinfer": {
                "tier": res.tier,
                "backend": res.backend,
                "route": res.route,
                "fell_back": res.fell_back,
                "reason": res.reason,
                "idempotency_key": res.idempotency_key,
                "latency_ms": round(res.latency_ms, 1),
                "ttft_ms": round(res.ttft_ms, 1) if res.ttft_ms is not None else None,
                "state": router.state(),
            },
        }

    return app


def serve(settings: Settings) -> None:
    import uvicorn

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)
