"""Command-line interface: `hybridinfer init | serve | run | models`."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import __version__
from .config import DEFAULT_CONFIG_YAML, Settings, load_settings

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.hybridinfer/config.yaml")


def _load(path: Optional[str]) -> Settings:
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        sys.exit(
            "No config at %s. Create one with:  hybridinfer init\n"
            "or pass --config <path>." % path
        )
    return load_settings(path)


def _cmd_init(args: argparse.Namespace) -> int:
    path = os.path.expanduser(args.path or DEFAULT_CONFIG_PATH)
    if os.path.exists(path) and not args.force:
        print("Config already exists at %s (use --force to overwrite)." % path)
        return 1
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(DEFAULT_CONFIG_YAML)
    print("Wrote %s" % path)
    print("Next: install/run Ollama, `ollama pull llama3.2:3b`, set OPENAI_API_KEY,")
    print("then `hybridinfer serve`.")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    settings = _load(args.config)
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    from .server import serve

    print("HybridInfer serving on http://%s:%d  (OpenAI-compatible)" % (settings.host, settings.port))
    print("Local:  %s" % (settings.local.model if settings.local else "-"))
    print("Remote: %s" % (settings.remote.model if settings.remote else "-"))
    serve(settings)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    settings = _load(args.config)
    from .router import HybridRouter

    router = HybridRouter(settings)
    prompt = args.prompt if args.prompt else sys.stdin.read()
    messages = [{"role": "user", "content": prompt}]

    if args.stream and not args.json:
        meta = {}
        for ch in router.stream(messages):
            if ch.done:
                meta = ch.meta or {}
                if ch.error:
                    meta = dict(meta)
                    meta["error"] = ch.error
                break
            sys.stdout.write(ch.delta)
            sys.stdout.flush()
        route = " -> ".join(meta.get("route", []))
        note = " (fell back)" if meta.get("fell_back") else ""
        err = meta.get("error")
        sys.stderr.write("\n[%s via %s%s%s]\n" % (
            meta.get("tier", "?"), route, note, (", error: %s" % err) if err else ""))
        return 0 if not err else 1

    res = router.complete(messages)
    if args.json:
        print(json.dumps(res.__dict__, indent=2))
        return 0 if res.ok else 1
    if not res.ok:
        sys.stderr.write("all tiers failed (last error: %s; route: %s)\n" % (res.error, res.route))
        return 1
    print(res.text)
    tag = " -> ".join(res.route)
    sys.stderr.write("\n[%s via %s%s, %.0f ms]\n" % (res.tier, tag, " (fell back)" if res.fell_back else "", res.latency_ms))
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    settings = _load(args.config)
    from .router import HybridRouter

    for m in HybridRouter(settings).models():
        print(m)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hybridinfer",
        description="Reliability-aware LLM router: local-first with automatic fallback.",
    )
    p.add_argument("--version", action="version", version="hybridinfer %s" % __version__)
    sub = p.add_subparsers(dest="command")

    pi = sub.add_parser("init", help="write a starter config file")
    pi.add_argument("path", nargs="?", help="config path (default ~/.hybridinfer/config.yaml)")
    pi.add_argument("--force", action="store_true", help="overwrite an existing config")
    pi.set_defaults(func=_cmd_init)

    ps = sub.add_parser("serve", help="start the OpenAI-compatible router server")
    ps.add_argument("--config", help="config path")
    ps.add_argument("--host")
    ps.add_argument("--port", type=int)
    ps.set_defaults(func=_cmd_serve)

    pr = sub.add_parser("run", help="route a single prompt and print the answer")
    pr.add_argument("prompt", nargs="?", help="prompt text (or read stdin)")
    pr.add_argument("--config", help="config path")
    pr.add_argument("--stream", action="store_true", help="stream tokens as they arrive")
    pr.add_argument("--json", action="store_true", help="print the full result as JSON")
    pr.set_defaults(func=_cmd_run)

    pm = sub.add_parser("models", help="list configured tiers/models")
    pm.add_argument("--config", help="config path")
    pm.set_defaults(func=_cmd_models)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
