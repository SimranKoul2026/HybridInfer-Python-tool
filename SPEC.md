# HybridInfer Design Specification (v1)

This is the **single source of truth** for HybridInfer's routing and reliability
behavior. There are two implementations of this one design:

- **`hybridinfer` (Python / PyPI)** - for laptops/desktops/servers; local tier = Ollama.
- **`com.hybridinfer` (Kotlin / Android AAR)** - for Android apps; local tier = MLC-LLM.

Both MUST implement the algorithms below identically. Behavioral parity of the
deterministic core is enforced by shared conformance vectors
(`conformance/vectors.json`), which both test suites load and must pass.

---

## 1. Concept

Route each request to a **local** tier first when it can safely be served there,
and fall back to a **remote** tier when local inference stalls, crashes, or is
*predicted* to fail. Reliability is a first-class routing signal, not an
afterthought.

Tiers:
- **local** - an on-device model (Ollama on desktop, MLC-LLM on Android).
- **remote** - any OpenAI-compatible endpoint.

The library/tool is a **router**, not an inference engine. It orchestrates
engines the host provides; it does not run model weights.

---

## 2. Complexity estimation

`estimate_tokens(text)` approximates token count without a tokenizer. It MUST be
computed identically in both implementations (round-half-up, integer result):

```
chars = length(text)
words = number of whitespace-separated tokens in text
raw   = max(chars / 4.0, words * 0.75)
estimate_tokens = max(1, floor(raw + 0.5))          # round half up, then floor
```

For an empty string, `estimate_tokens = 0`.

`prompt_tokens(messages)` = `estimate_tokens` of all message text concatenated
with newlines (string contents, plus the `text` fields of any content-part list).

Complexity bin (thresholds default: short_max=128, medium_max=512):
```
t = prompt_tokens(messages)
bin = 0 (short)   if t < short_max
      1 (medium)  if t < medium_max
      2 (long)    otherwise
```

---

## 3. Risk profile (self-calibrating)

Per-`(backend, model, length-bin)` failure-risk estimate, blending a prior with
observed outcomes (Beta-style).

```
PRIOR_FAIL   = [0.02, 0.10, 0.55]   # index = bin (short, medium, long)
PRIOR_WEIGHT = 8.0
key(backend, model, bin) = "backend|model|bin"

pr_fail(backend, model, bin):
    bin = clamp(bin, 0, 2)
    (attempts, failures) = stats[key] or (0, 0)
    num = PRIOR_FAIL[bin] * PRIOR_WEIGHT + failures
    den = PRIOR_WEIGHT + attempts
    return clamp(num / den, 0.0, 1.0)

update(backend, model, bin, failed):
    attempts += 1                       # capped at 10000
    if failed: failures += 1            # capped at 10000
    persist()
```

Persistence: a small JSON map `key -> [attempts, failures]`. A corrupt/missing
store MUST NOT crash the router; start empty.

---

## 4. Safety state machine

States: `LOCAL_ELIGIBLE`, `CAUTION`, `UNSAFE`, `RECOVERING`, `RESTORED`.
Parameters (defaults): `caution_pfail = 0.5`, `unsafe_failures = 2`,
`recovery_cooldown_s = 60`.

```
on_decision(pfail):                      # soft, pre-request gating
    if state in {LOCAL_ELIGIBLE, RESTORED} and pfail >= caution_pfail:
        state = CAUTION

local_allowed() -> bool:
    if state in {LOCAL_ELIGIBLE, CAUTION, RESTORED}: return true
    # UNSAFE / RECOVERING: allow ONE probe once cooldown elapses
    if unsafe_since != null and (now - unsafe_since) >= recovery_cooldown_s:
        state = RECOVERING
        return true
    return false

on_local_success():
    consec_fail = 0
    if state == RECOVERING:            state = RESTORED; unsafe_since = null
    elif state in {CAUTION, RESTORED}: state = LOCAL_ELIGIBLE

on_local_failure():
    consec_fail += 1
    if state == RECOVERING:            state = UNSAFE; unsafe_since = now   # probe failed
    elif consec_fail >= unsafe_failures: state = UNSAFE; unsafe_since = now
    else:                              state = CAUTION
```

`now` comes from an injectable monotonic clock (for testability).

---

## 5. Runtime-health monitor

Rolling counters over a window (default 300 s): recent failures, recent timeouts
(error in {timeout, stall}), last latency, last TTFT, consecutive-local count.
Informational for hosts/telemetry; the routing gate is driven by the risk profile
and state machine above. (Android MAY additionally fold in thermal headroom;
desktop has no portable thermal signal - see 9.)

---

## 6. Routing decision (`prefer_local`)

```
prefer_local(bin):
    if force_remote: return false
    if force_local:  return true
    if local is null:  return false
    if remote is null: return true
    if enable_runtime_health_gating:
        pfail = risk.pr_fail(local.backend, local.model, bin)
        state.on_decision(pfail)
        if pfail >= risk_prefer_remote: return false
    return true
```

`local_permitted()` = `enable_recovery ? state.local_allowed() : true`.

---

## 7. Fallback semantics

### Non-streaming
1. If `prefer_local` and local exists and `local_permitted()`: run local under a
   hard timeout + stall watchdog.
   - Success -> `on_local_success`, return local result.
   - Failure -> `on_local_failure`; if `enable_in_request_fallback` and remote
     exists, fall through to remote; else return the failure.
2. Remote: run remote, return (route records both tiers; `fell_back = true` if a
   local attempt preceded it).
3. Record every local outcome into the risk profile (`failed = not ok`).

### Streaming: first-token commit
A token, once sent, cannot be un-sent, so:
- If local fails **before** emitting any token (dominant prefill-wedge case),
  fall back to remote cleanly - the client sees only the remote stream.
- Once local emits a token, the request is **committed** to local; a mid-stream
  failure ends the stream with an error on the final event (no splicing a second
  model into a partial response).

### Failure codes
`timeout`, `stall` (token-rate collapse), `connection`, `oom`, `server_error`,
`empty`, or `http_<code>`.

---

## 8. Config schema (shared keys)

```
local  / remote : { backend, model, base_url, api_key(_env) }
routing:
  local_timeout_s, local_stall_timeout_s, remote_timeout_s,
  risk_prefer_remote, enable_runtime_health_gating,
  enable_in_request_fallback, enable_recovery, force_local, force_remote
complexity: short_max_tokens, medium_max_tokens
risk_profile_path
```

The feature flags reproduce the research A0-A3 ablation arms
(`force_local` no-control, runtime-health gating + fallback = runtime-only, all
on = full).

---

## 9. Platform differences (allowed to differ)

| Aspect | Python (desktop) | Kotlin (Android) |
|---|---|---|
| Local engine | Ollama (HTTP) | MLC-LLM (in-process) |
| Remote engine | OpenAI-compatible HTTP | OpenAI-compatible HTTP |
| Extra signal | none | thermal headroom (optional) |
| Persistence | JSON file | JSON file / SharedPreferences |
| Packaging | PyPI wheel | Maven/AAR |
| Surface | proxy server + CLI + lib | library (host builds UI) |

Everything in sections 2-8 MUST match regardless of platform.

---

## 10. Conformance

`conformance/vectors.json` is the shared, language-neutral test fixture. It is
byte-identical in both repos and covers: complexity binning, risk `pr_fail`,
safety-state transitions, and `prefer_local`. Both the Python and Kotlin test
suites load it and MUST pass every case. Changing behavior means changing the
vectors here first, then both implementations.
