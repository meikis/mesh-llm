# Pipelined VerifyWindow Decode

## Purpose

This document describes Skippy's internal speculative-decode subsystem for
native multi-token prediction (MTP) and the optional MTP-anchored N-gram
extender. It covers the wire protocol, target-verification invariant,
asynchronous scheduling, operating modes, and diagnostic telemetry.

The staged-runtime protocol deliberately has no compatibility path for the
retired synchronous `VerifySpan` message. Public mesh gossip and the
OpenAI-compatible API retain their normal compatibility guarantees.

## Terms

| Term | Meaning |
|---|---|
| Target | The full staged model, authoritative for every emitted token. |
| Native MTP | Model-provided typed draft attached to a target reply. GLM 4.7 Flash currently supplies a narrow `N+1` candidate. |
| N-gram sidecar | An upstream llama.cpp `ngram-simple` lookup or request-local `ngram-cache` proposer over target-committed tokens. |
| Composite proposal | Native-MTP prefix plus an optional N-gram suffix. |
| VerifyWindow | Versioned target request that verifies a candidate span at one session position. |
| Free target token | Target's next token after a fully verified span. |
| Stale window | Optimistic in-flight window invalidated by an earlier divergence. |

## Safety Invariant

MTP and N-gram are candidate sources, never authorities. The target verifies
each candidate sequence, and Skippy commits only the longest target-matching
prefix. A target correction is committed after a rejection.

```mermaid
flowchart LR
  MTP["Native MTP draft"] --> C["Composite candidate"]
  NGRAM["N-gram continuation"] --> C
  C --> W["VerifyWindow to target"]
  W --> R{"Target result"}
  R -->|"full accept"| A["Commit verified candidate\nand free target token"]
  R -->|"partial accept"| P["Commit matching prefix\nthen target correction"]
  R -->|"no proposal"| T["Ordinary target decode"]
```

This invariant also applies when multiple windows are in flight.

## Wire Protocol

`STAGE_STATE_VERSION` is `8`. `VerifyWindow` is wire message kind `21`; the
legacy kind `10` is rejected. An old/new staged-runtime pairing therefore fails
clearly instead of silently interpreting requests with different semantics.

```mermaid
sequenceDiagram
  participant S0 as "Stage 0 / OpenAI frontend"
  participant ST as "Downstream stages + target"
  Note over S0,ST: "Typed native MTP draft travels in target-reply sideband"
  S0->>ST: "VerifyWindow(id, position, current + candidates)"
  ST-->>S0: "PredictedTokens(window id, target tokens, next MTP draft)"
  S0->>S0: "Classify longest matching candidate prefix"
  S0->>S0: "Commit target-verified tokens only"
```

Pipelined decode requires direct prediction return. The target reply must reach
stage zero through the upstream-opened return sink; configuration fails when
that sink is unavailable.

## Composite Proposals

The sidecar extends native MTP; it never replaces it. Skippy uses upstream
llama.cpp proposers rather than a second Rust history scanner. `ngram-simple`
requires a historical continuation to begin with every MTP token before its
remaining tokens can become the sidecar tail. The request-local `ngram-cache`
instead reads directly after the provisional MTP prefix and returns only the
tail.

```mermaid
flowchart TD
  D["Receive typed MTP draft"] --> Q{"MTP tokens?"}
  Q -->|"no"| P["Pure N-gram fallback\nfrom accepted context"]
  Q -->|"yes"| H["Find latest earlier suffix match"]
  H --> A{"Historical continuation\nstarts with full MTP prefix?"}
  A -->|"yes"| X["Append useful suffix only"]
  A -->|"no"| M["Keep MTP prefix only"]
  P --> C["Composite proposal"]
  X --> C
  M --> C
  C --> V["One VerifyWindow"]
```

For MTP `[a, b]`, a valid historical continuation must start `[a, b, ...]`.
The composite proposal becomes `[a, b, c, d]`, not two independent requests.
A one-token N-gram tail is discarded. A rejected tail does not count as an MTP
prefix rejection; it only backs off the sidecar.

The cache is never shared between requests and is updated only after target
tokens commit. Drafting with `[a, b]` is read-only, so a rejected VerifyWindow
cannot affect a later lookup. This permits a cache tail to follow MTP even when
the cache would not independently predict `[a, b]`.

## Adaptive Sidecar Policy

The sidecar begins with the smallest useful tail. A fully accepted tail widens
the next tail by one token, up to the configured maximum. A rejected tail resets
the width and enters sidecar cooldown. With no MTP token, pure N-gram can use
the available N-gram budget.

```mermaid
stateDiagram-v2
  [*] --> InitialTail
  InitialTail --> WiderTail: "full tail accepted"
  WiderTail --> WiderTail: "full tail accepted below maximum"
  InitialTail --> Cooldown: "tail rejected"
  WiderTail --> Cooldown: "tail rejected"
  Cooldown --> NativeOnly: "cooldown proposal"
  NativeOnly --> Cooldown: "cooldown remains"
  NativeOnly --> InitialTail: "cooldown exhausted"
```

## Serial Native-MTP Mode

Native MTP alone uses serial VerifyWindow processing. A window is opened,
verified, classified, and committed before the next window begins.

```mermaid
sequenceDiagram
  participant F as "Frontend"
  participant T as "Target"
  F->>T: "Window 41: current + MTP candidate"
  T-->>F: "Window 41 reply"
  F->>F: "Commit verified prefix"
  F->>T: "Window 42: next candidate"
  T-->>F: "Window 42 reply"
```

This is the native-MTP parity path. It is not decode parallelism by itself.

## Pipelined Composite Mode

Concurrency is enabled only when the N-gram hybrid is on and
`SKIPPY_VERIFY_WINDOW_PIPELINE_DEPTH > 1`. A deeper composite proposal is
partitioned into FIFO windows. The target's free-advance candidate is reserved
as the next window's optimistic current token, preventing duplicate KV
positions.

```mermaid
sequenceDiagram
  participant F as "Stage-zero frontend"
  participant T as "Target"
  Note over F: "Composite proposal: [m1, n1, n2, n3]"
  F->>T: "Window 100 verifies [m1, n1]"
  F->>T: "Window 101 verifies [n2, n3]"
  T-->>F: "Window 100 reply"
  F->>F: "Commit verified prefix"
  T-->>F: "Window 101 reply"
  F->>F: "Commit only if prefix remains valid"
```

Replies complete in FIFO window-id order. An earlier divergence invalidates
later optimistic windows. Skippy drains them, records them as stale, trims to
the committed target state, and resumes from the target correction.

```mermaid
flowchart LR
  W0["Earlier window\npartial accept"] --> C["Commit matching prefix\n+ correction"]
  W0 --> D["Discard later windows\nas stale"]
  D --> R["Trim/replay to target state"]
  R --> N["Create candidate at corrected position"]
```

## Verification Outcomes

| Target result | Committed output | Next action |
|---|---|---|
| Full accept | Candidate plus free target token where applicable | Continue; adaptive width may grow. |
| Tail rejection | MTP prefix and matching tail prefix, then correction | Back off sidecar only. |
| Prefix rejection | Matching prefix, then correction | Handle native MTP rejection and discard stale windows. |
| EOG | Verified prefix through EOG | Stop. |
| No candidate | Ordinary target token | Continue decode. |

## Running On The Two-Host Lab

Use the package-qualified model reference. The normal mesh runtime owns split
planning; do not replace it with a direct `gguf://` reference for this flow.

### No MTP Baseline

```bash
SKIPPY_NATIVE_MTP_ENABLED=0 \
SKIPPY_NATIVE_MTP_NGRAM_HYBRID=0 \
SKIPPY_VERIFY_WINDOW_PIPELINE_DEPTH=1 \
mesh-llm serve meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M --split --no-draft
```

### Native MTP Only

```bash
SKIPPY_NATIVE_MTP_ENABLED=1 \
SKIPPY_NATIVE_MTP_NGRAM_HYBRID=0 \
SKIPPY_VERIFY_WINDOW_PIPELINE_DEPTH=1 \
mesh-llm serve meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M --split --no-draft
```

### MTP With Cache-backed N-gram Extension

```bash
SKIPPY_NATIVE_MTP_ENABLED=1 \
SKIPPY_NATIVE_MTP_NGRAM_HYBRID=1 \
SKIPPY_NGRAM_CACHE_ENABLED=1 \
SKIPPY_NGRAM_CACHE_MIN_NGRAM=2 \
SKIPPY_NGRAM_CACHE_MAX_NGRAM=4 \
SKIPPY_NATIVE_MTP_NGRAM_SIZE=4 \
SKIPPY_NATIVE_MTP_NGRAM_MAX_PROPOSAL_TOKENS=4 \
SKIPPY_NATIVE_MTP_NGRAM_TAIL_BACKOFF_PROPOSALS=6 \
SKIPPY_NATIVE_MTP_VERIFY_WINDOW_MIN_TOKENS=1 \
SKIPPY_NATIVE_MTP_VERIFY_WINDOW_MAX_TOKENS=4 \
SKIPPY_VERIFY_WINDOW_PIPELINE_DEPTH=2 \
mesh-llm serve meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M --split --no-draft
```

```mermaid
flowchart LR
  C["SPEED-Bench client\non micstudio"] --> S0["micstudio :9337\nOpenAI frontend / stage 0"]
  S0 --> L["Persistent direct-LAN\nbinary stage lanes"]
  L --> S1["studio54\nstage 1"]
  S1 --> S0
  S0 --> C
```

The normal planner currently selected `micstudio 0..47` and `studio54 47..48`.
Record layer ranges, direct RTT, lane count, context size, and binary commit
with every benchmark. That shape proves normal split serving but is not directly
comparable to historic 22/26 benchmark rows.

## Telemetry And Interpretation

The OpenAI response `timings` object provides aggregate evidence; debug
telemetry supplies per-window and per-stage detail.

| Question | Counters |
|---|---|
| Is decode faster? | `predicted_per_second`, `predicted_n`, `predicted_ms` |
| Are proposals accepted? | `draft_n`, `draft_n_accepted` |
| Did the sidecar widen MTP? | `native_mtp_hybrid_native_tokens`, `native_mtp_hybrid_ngram_tokens`, `native_mtp_hybrid_proposed_tokens` |
| Did anchors agree? | `native_mtp_hybrid_ngram_mtp_prefix_agreements`, `native_mtp_hybrid_ngram_mtp_prefix_disagreements` |
| Were tails useful? | `native_mtp_hybrid_accepted_tail_tokens`, `native_mtp_hybrid_ngram_tail_rejections`, `native_mtp_hybrid_ngram_sidecar_backoffs` |
| Was it pipelined? | `verify_window_depth`, `verify_window_opened`, `verify_window_max_in_flight`, `verify_window_stale_discarded` |
| Where was time spent? | `verify_window_downstream_wait_ms`, `verify_window_forward_write_ms`, `verify_window_stage0_compute_ms`, `verify_window_verify_elapsed_ms` |

A useful hybrid run needs more than an increased `draft_n`: it needs accepted
tail tokens, high anchor agreement, bounded stale-window work, and completion
throughput higher than the native-MTP control.
