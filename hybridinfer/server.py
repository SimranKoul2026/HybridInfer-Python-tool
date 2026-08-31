"""OpenAI-compatible HTTP server.

Exposes POST /v1/chat/completions so any OpenAI client (or app) can point at
HybridInfer and get transparent local-first routing with automatic fallback.
Also serves /v1/models, /healthz, and /stats.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, List

from .config import Settings
from .router import HybridRouter


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


def _event_stream(router: HybridRouter, messages: List[Dict[str, Any]]) -> Iterator[str]:
    """Format the controller's StreamChunks as OpenAI-style SSE."""
    created = int(time.time())
    cid = "hybridinfer-%d" % created
    model = "hybridinfer"
    role_sent = False
    meta: Dict[str, Any] = {}
    for ch in router.stream(messages):
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
    # final chunk carries finish_reason + hybridinfer routing metadata
    yield _sse_chunk(cid, created, model, {}, "stop", extra=meta)
    yield "data: [DONE]\n\n"


def create_app(settings: Settings):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="HybridInfer", version="0.1.0")
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
    async def chat_completions(body: Dict[str, Any]) -> Any:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "`messages` is required", "type": "invalid_request_error"}},
            )

        # Streaming: return SSE. Starlette iterates the sync generator in a
        # threadpool, so the blocking backend HTTP stays off the event loop.
        if body.get("stream"):
            return StreamingResponse(
                _event_stream(router, messages),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # The router does blocking HTTP; run it off the event loop.
        import anyio

        res = await anyio.to_thread.run_sync(router.complete, messages)

        if not res.ok:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "all tiers failed (last error: %s)" % res.error,
                        "type": "upstream_error",
                        "hybridinfer": {"route": res.route, "error": res.error},
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
