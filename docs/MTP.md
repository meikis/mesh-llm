# MTP and Draft Speculation

This document records the current decision: Skippy no longer carries its own
MTP implementation. The upstream llama.cpp MTP path owns same-model MTP support.
Skippy keeps only draft-model speculation work.

## Decision

- Delete Skippy-owned MTP ABI, CLI flags, staged protocol, package strategy
  metadata, and Rust orchestration.
- Keep draft-model speculation:
  - single-stage native slot `draft-simple` through llama.cpp;
  - multi-stage stage0 draft model proposals verified by downstream
    `SpeculativeSpan` messages.
- Do not put benchmark results in `model-package.json`.
- Package speculative metadata may describe `draft-model` or `ngram`
  strategies only.

## Why

The hand-rolled Skippy MTP path duplicated llama.cpp speculation machinery and
kept drifting from the optimized server slot loop. That made performance and
correctness hard to reason about. The useful Skippy work is draft-model
speculation in distributed topologies, where stage0 can run a small draft model
and ask the target pipeline to verify proposed spans.

## Current Boundary

Upstream llama.cpp:

- owns same-GGUF MTP and any MTP-specific recurrent-state handling;
- can be benchmarked directly with `llama-server` when we want MTP data.

Skippy:

- owns split serving, stage transport, telemetry, and draft-model speculative
  verification;
- does not expose `--mtp`, `draft-mtp`, `final-stage-head`, or
  `DecodeReadoutMtp`;
- sends draft proposals with `SpeculativeSpan`.

## Benchmark Baseline

Historical MTP and draft-spec benchmark outputs remain under
`experiments/mtp/`. Treat them as experiment artifacts, not package metadata.
New Skippy speculation runs should compare:

1. vanilla target baseline;
2. vanilla target plus draft model;
3. Skippy single-stage baseline;
4. Skippy single-stage plus draft model;
5. Skippy multi-stage baseline;
6. Skippy multi-stage plus draft model.

Use the scripted harnesses so source sync, warmup, cooldown, labels, and metrics
collection stay repeatable.

## Target PR Direction

The next review-ready PR should make package-declared draft speculation the
normal path instead of an experiment-only flag set. The target scope is:

- enable draft speculation from `model-package.json` when a package declares a
  default draft strategy;
- add or publish a layer package repository for the known-winning Llama 3.3 70B
  Q3 target plus Llama 3.2 1B Q4 draft pair;
- remove obsolete speculation experiments and cruft that are not part of
  package-declared `draft-model` speculation;
- sync with the main repository before review;
- keep benchmark evidence in docs and experiment JSON, not in the package
  manifest.

The package manifest should stay small and declarative. For the winning pair,
the intended shape is:

```json
{
  "generation": {
    "speculative_decoding": {
      "default": "llama32-1b-q4",
      "strategies": {
        "llama32-1b-q4": {
          "type": "draft-model",
          "draft_model": "unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M@<revision>",
          "window_policy": {
            "default": "adaptive",
            "initial_window": 16,
            "min_window": 2,
            "max_window": 16
          }
        }
      }
    }
  }
}
```

Important manifest decisions:

- `draft_model` is a normal model resolver ref. It should download through the
  same model cache/resolver path as other models.
- The strategy's presence and selection as `default` are the compatibility
  signal. Do not add a redundant `compatible: true`.
- Adaptive is the default policy, but it starts at the proven W16 setting for
  this pair.
- Do not put fallback thresholds, cost breakers, lower-prefetch knobs, or
  benchmark scores in `model-package.json`; those remain runtime policy.
- If the draft model cannot be resolved or the user disables speculation, serve
  the target package normally without failing model load.

Runtime/user control should stay separate from the package manifest. The package
declares available strategies; the existing speculative runtime config decides
whether to use them:

```toml
[defaults.speculative]
mode = "auto"      # auto | disabled | draft | ngram
package_strategy = "default"
```

Serving behavior should stay simple:

- `auto` may use the package default strategy when it is available and
  resolvable;
- `disabled` never uses package-declared speculation;
- `package_strategy = "default"` uses the manifest default, while another
  value selects a named package strategy;
- explicit `draft` mode still uses the normal manually configured draft path.

There should be no `required` draft mode. In `auto`, missing or unresolved draft
models fall back to baseline serving.

## Two-Stage Draft Prefetch

The cross-host stage0 draft harness enables lower-stage prefetch by default for
stage0 draft speculation. For hybrid recurrent models, useful lower prefetch
requires enough recurrent rollback snapshots for both the current verify span
and the next prefetched span. The harness now derives that automatically:

- baseline keeps the requested `N_RS_SEQ` value, which defaults to `0`;
- draft W2 needs `n_rs_seq >= 3`;
- draft W3 needs `n_rs_seq >= 5`;
- in general, fixed window `W` with lower prefetch needs
  `n_rs_seq >= 2 * W - 1`.

That rule exists because llama.cpp exposes a rollback horizon of
`n_rs_seq + 1` tokens. Without the larger horizon, stage0 can verify the current
span but cannot safely keep a prefetched lower-stage result for the next span.

Smoke evidence from
`experiments/mtp/crosshost/two-stage-lower-prefetch-auto-20260524T202515Z`:

| Condition | Stage0 `n_rs_seq` | Lower prefetch useful | Tok/s | Speedup |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0 | n/a | 9.27 | 1.00x |
| draft W2 | 3 | 27 | 12.05 | 1.30x |
| draft W3 | 5 | 15 | 9.66 | 1.04x |

The W2 run matched the baseline output hash and had no replay commits, so the
prefetch work was both useful and correctness-preserving in this smoke.

## Draft Guard Finding

Full-corpus W2 draft speculation needs a request-local fallback guard. The
one-prompt smoke had high acceptance, but the 9-prompt corpus includes
low-acceptance spans, especially the long code review prompt. Unguarded W2 with
lower prefetch was unstable:

| Run | Tok/s |
| --- | ---: |
| split60 baseline | 9.29 |
| unguarded W2 attempt 1 | 6.27 |
| unguarded W2 attempt 2 | 9.75 |
| unguarded W2 attempt 3 | 7.82 |

Skippy now disables stage0 draft speculation for the rest of a request once the
request has at least `SKIPPY_STAGE0_SPEC_FALLBACK_MIN_WINDOWS` windows and the
accepted/proposed ratio falls below
`SKIPPY_STAGE0_SPEC_FALLBACK_MIN_ACCEPT_RATE`. Defaults are `32` windows and
`0.90`. If lower prefetch already advanced stage0 when the guard trips, Skippy
forces the verified-prefix commit path before continuing with normal one-token
split decode.

Guarded W2 confirmation from
`experiments/mtp/crosshost/two-stage-w2-guard-confirm-20260524T214529Z`:

| Condition | Attempts tok/s | Median tok/s | Speedup vs split baseline |
| --- | --- | ---: | ---: |
| split60 baseline | 9.32, 9.29, 9.29 | 9.29 | 1.00x |
| W2 guarded lower prefetch | 8.17, 9.74, 9.50 | 9.50 | 1.02x |

Controls:

- `SPEC_DRAFT_P_MIN=0.90` was worse (`8.40 tok/s`).
- W3 guarded was worse (`7.78 tok/s`).
- Disabling lower prefetch was worse (`8.09 tok/s`).
- Split62 was worse for both baseline (`9.15 tok/s`) and W2 (`7.78 tok/s`).
- Disabling the lower-prefetch cooldown was better in a one-run check
  (`9.82 tok/s`), so the default cooldown is `0`; the fallback guard is the
  safety valve for low-acceptance prompts.
- The default lower-prefetch policy is now aggressive: when lower prefetch is
  enabled and the recurrent snapshot horizon allows the next span, Skippy
  schedules it instead of using the EWMA net-benefit gate. The old cost gate is
  still available for experiments with
  `SKIPPY_STAGE0_LOWER_PREFETCH_COST_GATE=1`.
- Aggressive lower prefetch confirmation from
  `experiments/mtp/crosshost/two-stage-w2-aggressive-prefetch-confirm-20260524T221403Z`
  scheduled all `611/611` lower-prefetch candidates and reduced measured
  downstream wait to roughly `0.5s` on the two valid attempts. Those attempts
  were `10.18 tok/s` and `10.18 tok/s`, or about `1.10x` over the split60
  baseline. The third attempt was a slow outlier at `7.20 tok/s`; its downstream
  wait stayed low, but lower-verify and lower-prefetch time rose from about
  `56s` to about `97s`, so the remaining instability is exposed stage0 verify
  cost rather than missed prefetch.
- Skippy now keeps aggressive prefetch as the default and adds a lower-prefetch
  cost breaker. It compares prefetched lower-verify cost against a warmed
  stage0 verify baseline shared by requests in the same server process and
  disables lower prefetch for the current request after repeated slow spans. If
  the breaker fires after lower prefetch has advanced stage0, Skippy forces the
  verified-prefix commit path before continuing. The draft fallback guard
  remains separate and can still disable draft speculation for low-acceptance
  requests.
- Process-baseline breaker confirmation from
  `experiments/mtp/crosshost/two-stage-w2-process-breaker-confirm-20260524T225636Z`
  produced `10.18`, `10.23`, and `9.37 tok/s`, all valid, with median
  `10.18 tok/s`. That is about `1.10x` over the split60 baseline. Lower
  prefetch stayed fully scheduled (`611/611`) and downstream wait stayed low
  (`0.40s` to `0.54s`). The breaker did not fire in this run because the
  measured lower-prefetch cost never exceeded the warmed baseline threshold; the
  third run was slower because exposed lower-prefetch time rose to about
  `10.5s`, not because RTT waits returned.
- Skippy also has a request-local exposed-cost budget for lower prefetch. It
  tracks cumulative exposed lower-prefetch milliseconds per committed input and
  throttles lower prefetch for the rest of the request once the budget is
  exceeded. The default gate is enabled with
  `SKIPPY_STAGE0_LOWER_PREFETCH_EXPOSED_BUDGET=1`,
  `SKIPPY_STAGE0_LOWER_PREFETCH_EXPOSED_BUDGET_MIN_SPANS=32`,
  `SKIPPY_STAGE0_LOWER_PREFETCH_EXPOSED_BUDGET_MIN_MS=1000`, and
  `SKIPPY_STAGE0_LOWER_PREFETCH_EXPOSED_BUDGET_MS_PER_TOKEN=8.0`. Once tripped,
  lower prefetch is attempted every
  `SKIPPY_STAGE0_LOWER_PREFETCH_EXPOSED_BUDGET_THROTTLE_STRIDE=4` eligible
  spans instead of being disabled completely.
- Exposed-budget confirmation from
  `experiments/mtp/crosshost/two-stage-w2-exposed-budget-confirm-20260524T231848Z`
  produced two clean attempts (`10.22`, `10.20 tok/s`) and one rejected slow
  outlier (`7.79 tok/s`). The rejected attempt's short warmup was already bad
  (`1.50 tok/s` vs the normal `~12 tok/s`), so this was a poisoned fresh process
  before measurement rather than just a lower-prefetch policy miss.
- The cross-host harness now supports warmup health gating before measurement:
  `MEASURE_ATTEMPTS` can exceed `MEASURE_RUNS`, and attempts whose short warmup
  falls below `WARMUP_MIN_TOK_S` are restarted after cooldown. The first gated W2
  confirmation from
  `experiments/mtp/crosshost/two-stage-w2-warmup-gated-confirm-20260524T233715Z`
  used `MEASURE_RUNS=3`, `MEASURE_ATTEMPTS=5`, and `WARMUP_MIN_TOK_S=6.0`. It
  accepted three attempts with healthy warmups (`12.35`, `12.41`, `11.99 tok/s`)
  and measured `10.22`, `10.17`, and `9.35 tok/s`, median `10.17 tok/s`.
- Adaptive lower-prefetch throttle confirmation from
  `experiments/mtp/crosshost/two-stage-w2-throttle-confirm-20260524T235913Z`
  used the same warmup-gated harness. Attempt 3 was rejected before measurement
  because the 32-token warmup was only `2.12 tok/s`. The three measured attempts
  were `10.18`, `10.21`, and `10.16 tok/s`, median `10.18 tok/s`; no measured
  request tripped the exposed-budget throttle, so the change did not hurt the
  fast path. Metrics reports now include a `request_speculation` section keyed by
  request/session id for per-request stage0 speculation timing.

The remaining bottleneck is exposed draft/lower-verify cost and its variance,
not rollback correctness. W2 lower prefetch is useful, but the current two-stage
draft loop does not yet recover enough of the split penalty to approach vanilla
llama-server throughput.

## Same-Host Two-Stage Window Finding

For the Llama 3.3 70B Q3 target with the Llama 3.2 1B Q4 draft model on
Studio, draft speculation now wins over the two-stage baseline when stage0
lower prefetch is enabled. W8 established the first stable win; a follow-up
window sweep found W16 to be the current best same-host setting for this pair.

Key implementation decisions:

- Stage0 draft sessions trim away a p-min-rejected candidate token instead of
  leaving the draft KV advanced through a token that was never proposed.
- Stage0 lower prefetch is enabled by default. It can be disabled with
  `SKIPPY_STAGE0_LOWER_PREFETCH=0`.
- Draft p-min defaults to `0.60` for Skippy draft speculation. The CLI flags
  `--spec-draft-p-min` and `--openai-spec-draft-p-min` still override it for
  other model pairs.

Clean same-host Studio measurements, 2 prompts x 192 generated tokens:

| Condition | Median tok/s | Median wall | Notes |
| --- | ---: | ---: | --- |
| two-stage baseline | 7.957 | 48.26s | no speculation |
| W8, p-min 0.75, lower prefetch off | 8.134 | 47.21s | +2.2% vs baseline |
| W8, p-min 0.75, lower prefetch on | 8.447 | 45.46s | +6.2% vs baseline |
| W8, p-min 0.70, lower prefetch on | 8.449 | 45.45s | effectively tied with 0.75 |
| W8, p-min 0.60, lower prefetch on | 8.708 | 44.10s | +9.4% vs baseline |
| W16, p-min 0.60, lower prefetch on | 9.835 | 39.04s | +23.6% vs baseline |

The p-min 0.60 run accepted about `266.7` of `289.7` proposed tokens
(`~92.1%`) and committed about `3.31` inputs per speculative span. It is faster
than p-min 0.70/0.75 because the larger useful spans amortize more downstream
round trips, even though the lower-verify work is larger.

W16 confirmation from
`experiments/mtp/studio54/skippy-2stage-w16-defaultpmin-defaultlower-192tok-prompt2-3x-20260526T214535Z`
ran three clean attempts at `9.832`, `9.835`, and `10.002 tok/s`. It accepted
about `269.7` of `290.3` proposed tokens (`~92.9%`) and committed about `3.75`
inputs per speculative span. Lower-prefetch waste was low (`0.33` spans/run),
and downstream wait fell to about `30.5 ms` per committed input.

One-shot window sweep evidence:

| Window | Tok/s | Wall | Accepted / proposed | Committed inputs/span | Notes |
| ---: | ---: | ---: | ---: | ---: | --- |
| W6 | 8.963 | 42.84s | 270 / 287 | 3.14 | stable but smaller spans |
| W8 | 8.822 | 43.53s | 267 / 289 | 3.42 | below confirmed W8 3-run median |
| W10 | 9.284 | 41.36s | 269 / 291 | 3.36 | improvement over W8 |
| W12 | 9.668 | 39.72s | 271 / 291 | 3.76 | near plateau |
| W14 | 9.680 | 39.67s | 269 / 290 | 3.64 | near plateau |
| W16 | 9.845 | 39.01s | 269 / 290 | 3.68 | best one-shot and confirmed by 3-run pass |
| W20 | 3.699 | 103.82s | 269 / 290 | 3.84 | pathological exposed verify/wait outlier |
| W24 | 9.235 | 41.58s | 271 / 291 | 4.11 | recovers, but slower than W16 |
| adaptive W16 | 7.604 | 50.50s | 251 / 275 | 1.96 | current adaptive ramp is worse than baseline |

Do not treat larger windows as automatically better. W20 showed that a wider
window can preserve acceptance while still losing badly if lower verification
or downstream wait becomes exposed. For this pair, W16 is the current best
claim; future tuning should re-confirm around the plateau rather than pushing
window size blindly upward.

The old adaptive window policy was not a safe replacement for fixed W16 on this
benchmark. It spent too much of the short request at small spans (`~1.96`
committed inputs/span) and exposed more lower verify/downstream wait per
committed input. Stage0 adaptive draft now starts at the configured max window
by default, so `--openai-adaptive-speculative-window --openai-spec-draft-window
16` begins at W16 instead of slowly ramping from W2. Set
`SKIPPY_STAGE0_ADAPTIVE_START_MAX=0` to restore the old ramp-from-small
behavior for experiments. A same-host smoke after this change produced
`10.018 tok/s` and `38.33s` wall time at adaptive W16, back in the fixed-W16
range.

Clean apples-to-apples rerun from
`experiments/mtp/studio54/clean-table-llama33q3-draft1b-w16-20260527` used the
same source state on Studio, the same Llama 3.3 70B Q3 target, the same Llama
3.2 1B Q4 draft, the same first 2 PR prompts, `192` generated tokens,
temperature `0`, one warmup per attempt, `120s` cooldowns, and three clean
measurement attempts per condition.

| Runtime | Condition | Window | Median tok/s | Min-Max tok/s | Median wall | Acceptance | Valid | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama-server | baseline, no draft | - | 8.017 | 8.015-8.018 | 47.90s | n/a | 3/3 | 1.00x |
| llama-server | draft model speculation | W16 | 11.041 | 11.041-11.044 | 34.78s | 0.870 | 3/3 | 1.38x |
| skippy-server two-stage | baseline, no draft | - | 7.961 | 7.959-7.964 | 48.23s | n/a | 3/3 | 1.00x |
| skippy-server two-stage | stage0-owned draft speculation | W16 | 10.024 | 9.853-10.025 | 38.31s | 0.899 | 3/3 | 1.26x |
| skippy-server two-stage | adaptive stage0-owned draft speculation | W16 max | 10.020 | 9.867-10.036 | 38.33s | 0.899 | 3/3 | 1.26x |

This is a real Skippy win: fixed W16 draft speculation improves same-host
two-stage Skippy throughput by about `25.9%` and reduces median wall time by
about `20.6%` versus the matching two-stage Skippy baseline. Vanilla
llama-server remains the performance reference for this pair: Skippy two-stage
W16 is still about `9.2%` behind vanilla W16, even though Skippy baseline is
already close to vanilla baseline.
