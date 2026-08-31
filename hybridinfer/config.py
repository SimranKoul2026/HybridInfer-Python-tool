"""Configuration: dataclasses + a YAML/env loader.

Kept import-light (no yaml needed unless you actually load a file) so the core
package imports without optional deps.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BackendSpec:
    backend: str                       # "ollama" | "openai"
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None      # literal key (discouraged; prefer api_key_env)
    api_key_env: Optional[str] = None  # env var name to read the key from

    def resolved_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class Settings:
    local: Optional[BackendSpec] = None
    remote: Optional[BackendSpec] = None

    # controller / reliability
    local_timeout_s: float = 120.0
    local_stall_timeout_s: float = 20.0
    remote_timeout_s: float = 120.0
    risk_prefer_remote: float = 0.6
    enable_runtime_health_gating: bool = True
    enable_in_request_fallback: bool = True
    enable_recovery: bool = True
    force_local: bool = False
    force_remote: bool = False

    # complexity bins
    short_max_tokens: int = 128
    medium_max_tokens: int = 512

    # persistence
    risk_profile_path: Optional[str] = None

    # server
    host: str = "127.0.0.1"
    port: int = 8080


def _spec_from_dict(d: Optional[Dict[str, Any]]) -> Optional[BackendSpec]:
    if not d:
        return None
    return BackendSpec(
        backend=d["backend"],
        model=d["model"],
        base_url=d.get("base_url"),
        api_key=d.get("api_key"),
        api_key_env=d.get("api_key_env"),
    )


def settings_from_dict(data: Dict[str, Any]) -> Settings:
    routing = data.get("routing", {}) or {}
    complexity = data.get("complexity", {}) or {}
    server = data.get("server", {}) or {}
    return Settings(
        local=_spec_from_dict(data.get("local")),
        remote=_spec_from_dict(data.get("remote")),
        local_timeout_s=float(routing.get("local_timeout_s", 120.0)),
        local_stall_timeout_s=float(routing.get("local_stall_timeout_s", 20.0)),
        remote_timeout_s=float(routing.get("remote_timeout_s", 120.0)),
        risk_prefer_remote=float(routing.get("risk_prefer_remote", 0.6)),
        enable_runtime_health_gating=bool(routing.get("enable_runtime_health_gating", True)),
        enable_in_request_fallback=bool(routing.get("enable_in_request_fallback", True)),
        enable_recovery=bool(routing.get("enable_recovery", True)),
        force_local=bool(routing.get("force_local", False)),
        force_remote=bool(routing.get("force_remote", False)),
        short_max_tokens=int(complexity.get("short_max_tokens", 128)),
        medium_max_tokens=int(complexity.get("medium_max_tokens", 512)),
        risk_profile_path=data.get("risk_profile_path"),
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 8080)),
    )


def load_settings(path: str) -> Settings:
    """Load settings from a YAML file (requires pyyaml)."""
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Reading a config file needs PyYAML. Install with: pip install pyyaml"
        ) from e
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return settings_from_dict(data)


DEFAULT_CONFIG_YAML = """\
# HybridInfer configuration.
# Runs requests on your local model first and falls back to a remote model when
# local inference stalls, crashes, or is predicted to fail.

# --- Local tier: your on-device model, served by Ollama (https://ollama.com) ---
local:
  backend: ollama
  model: llama3.2:3b            # any model you've `ollama pull`ed
  base_url: http://127.0.0.1:11434

# --- Remote tier: any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM...) ---
remote:
  backend: openai
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
  api_key_env: OPENAI_API_KEY   # key is read from this environment variable

routing:
  local_timeout_s: 120          # hard ceiling for a local request
  local_stall_timeout_s: 20     # no new token for this long => treat local as wedged
  remote_timeout_s: 120
  risk_prefer_remote: 0.6       # predicted local-failure prob at/above which we skip local
  enable_runtime_health_gating: true   # learn per-model risk and pre-empt likely failures
  enable_in_request_fallback: true     # local fails mid-request => auto-retry on remote
  enable_recovery: true                # hold a wedged local tier out, then probe it back in
  force_local: false            # debug: always try local
  force_remote: false           # debug: always use remote

complexity:
  short_max_tokens: 128
  medium_max_tokens: 512

# Where the self-calibrating risk profile is persisted (survives restarts).
risk_profile_path: ~/.hybridinfer/risk_profile.json

server:
  host: 127.0.0.1
  port: 8080
"""
