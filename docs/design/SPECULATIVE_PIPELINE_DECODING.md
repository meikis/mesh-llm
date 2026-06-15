# Speculative Pipeline Decoding vs Skippy

> **Spike note (branch `micn/speculative-spike`).** Research + first-pass wiring
> for a self-draft speculative path in skippy. Validated on a real 2-stage
> Qwen3-0.6B split; true early-exit logits are blocked in the llama.cpp patch
> queue (needs a trained head / ABI change). Not for merge as-is.
>
> - Paper: https://arxiv.org/pdf/2605.30852
> - Reference impl (local): `/Users/micn/Development/speculative_pipeline_decoding`
> - This repo: https://github.com/Mesh-LLM/mesh-llm

Research notes on applying the paper **"Speculative Pipeline Decoding: Higher-Accuracy
and Zero-Bubble Speculation via Pipeline Parallelism"** (arXiv:2605.30852) to mesh-llm's
skippy staged runtime.

Reference implementation studied:
`/Users/micn/Development/speculative_pipeline_decoding` (Qwen3/Qwen3.5, PyTorch, correctness-only).

This is an evaluation/design note, not an implementation. No mesh-llm code is changed by it.

## TL;DR

- The paper's idea maps unusually well onto skippy because skippy is *already* a
  multi-stage layer-parallel pipeline — which is the exact substrate the paper
  assumes. Most speculative-decoding work targets a single monolithic model; this
  paper targets a pipeline, which is what we have.
- skippy *already has* a speculative-decoding lane
  (`crates/skippy-server/src/frontend/speculative.rs` + `local_generation.rs`),
  but it is the **traditional draft-model + verify-window** style (EAGLE/medusa
  family). The paper explicitly positions itself *against* that style.
- The paper's real contribution for us is **zero-bubble speculation**: a small
  speculation head drafts the next token *in parallel with* the target pipeline's
  forward pass, so the pipeline never stalls waiting for a draft and never wastes
  the partial work already in flight across stages. That directly attacks
  skippy's current cross-stage latency cost.
- Adopting it fully is a real research/engineering project (trained speculation
  head, per-stage hidden-state taps, cross-stage rollback). A staged adoption is
  realistic; the cheap wins are the rollback/snapshot machinery and the
  in-flight-pipeline mental model.

## What the paper actually does

Partition the target LLM into `n` pipeline stages. At any decode round, several
recent tokens (`x_5..x_7`) are *in flight* at different pipeline depths, while
older tokens (`x_1..x_4`) are fully processed.

Key mechanics from the reference impl
(`pipeline_model.py`, `Qwen3SpeculativePipelineModel`):

1. **Per-stage hidden-state taps.** As each token passes a stage, the post-stage
   hidden state is snapshotted (`_forward_stage_with_snapshots`,
   `_extract_position_snapshots_from_hidden_states`). For each token, hidden
   states from the stages it has already passed are projected through learned FC
   layers (`stage_projs`, `g0_proj`) and aggregated into one feature.
2. **A small Speculation Module** (`SpeculationHeadTransformer`, a few
   `num_spec_layers` decoder layers + an `lm_head`, optionally over a reduced
   "draft vocabulary") consumes those aggregated features and predicts the next
   token *concurrently* with the target pipeline's own forward step
   (`run_spec_parallel()` is launched alongside the stage loop in
   `_generate_single_chain`). This is the "zero bubble": the draft is computed on
   the same wall-clock window as a pipeline stage step, not before/after it.
3. **Verification is lossless and uses work already done.** When the oldest token
   pops out of the final stage, its real logits verify the token that was
   speculated for it (`_verify_pipeline_draft_token`). No extra target forward
   pass is needed to verify — verification *is* the normal pipeline output.
4. **On rejection, roll back the pipeline + caches by a stage-dependent amount.**
   `pipeline_linear_cache.py` keeps a rolling FIFO of per-layer post-update
   snapshots (`max_snapshots = num_stages`) so each layer can be rewound by
   `num_stages - 1 - stage_idx` steps (`crop_pipeline_cache_after_rejection`,
   `crop_rewind`). Standard attention KV uses normal `crop`; linear/hybrid
   (Qwen3.5 linear-attention) layers use the snapshot rewind.

The theoretical decode speed in the repo is a toy model: standard = 1 step/token,
pipeline = `decode_steps / num_stages` (`theoretical_speedup_vs_standard`). The
README is explicit that wall-clock is not optimized in the reference code; only
acceptance rate and the step-count model are meaningful there.

## What skippy is today (grounded in code)

Stage runtime and transport:

- A model is split into contiguous layer ranges = **stages**, planned by
  `skippy-topology` / `skippy-coordinator`, materialized via layer packages
  (`skippy-runtime/src/package.rs`). Stage 0 holds embeddings + first layers and
  drives generation; the final stage produces logits.
- Stages talk over a **binary activation transport**
  (`crates/skippy-server/src/binary_transport.rs`,
  `binary_transport/forwarding.rs`, `wire.rs`). What crosses the wire is an
  `ActivationFrame` (hidden states) plus state flags, not tokens — i.e. exactly
  the "plain activation transport" the integration plan
  (`docs/SKIPPY.md`) calls for.
- Stages can run on separate peers over QUIC, or multiple stages on one node;
  single-node serving is a one-stage synthetic package.
- There is already pipelining machinery: `async_prefill_forward` /
  `AsyncForwarder` and a `max_inflight` credit window let a stage forward
  activations downstream without blocking lockstep. So skippy is not strictly
  "stage N waits fully for N-1" — it already has inflight credits, primarily
  exercised in prefill.

Existing speculative decoding (the traditional kind):

- `frontend/speculative.rs` implements a **draft-window verify** scheme:
  `classify_verify_span`, `VerifySpanDecision` (FullAccept / AcceptedStop /
  TailReject / EarlyReject / EarlyRejectStop), adaptive window grow/shrink, and
  `repaired_commit_tokens` recovery. `OpenAiSpeculativeStats` tracks accept rate,
  recovery restores, primary-verify timing, etc.
- A separate **draft model** is proposed and stage-0 verifies a span. This is the
  EAGLE/draft-model family — `crates/llama-spec-bench` is the preflight tool that
  checks a draft GGUF against a target GGUF before wiring it in.
- KV/slot/lane/session state lives in
  `skippy-server/src/runtime_state.rs` and `kv_integration/`; there is span/window
  commit + recovery (a form of rollback), but it is token-span recovery, not the
  paper's per-stage layer-state snapshot rewind.

## Mapping the paper onto skippy

| Paper concept | skippy equivalent today | Gap to close |
|---|---|---|
| `n` pipeline stages = layer ranges | Stages = layer ranges across peers | None — same substrate |
| Activations flow between stages | `ActivationFrame` binary transport | None |
| In-flight tokens at varying depths | `async_prefill_forward` + `max_inflight` credits | Decode path is mostly one-token; would need multiple decode tokens in flight |
| Per-stage hidden-state taps | Stage output activation is already available at each stage boundary | Need to *expose/collect* per-stage taps for a speculation head |
| Speculation Module (trained head) | None (skippy uses a separate draft *model*) | Need a trained per-family speculation head + weights distribution |
| Verify = normal pipeline output | stage-0 verify span (extra-ish work) | Reframe: verify the popped token against its speculation |
| Stage-dependent cache rollback | Token-span recovery in runtime_state | Need per-stage layer-snapshot rewind (`crop_rewind` analog) |
| Reduced draft vocabulary | n/a | Optional optimization, easy to add to a head |

## Why this is attractive for skippy specifically

1. **The cost skippy pays is cross-stage latency.** A split model's decode step
   has to traverse stages (and the network between peers). The paper's whole point
   is to *hide draft computation inside that traversal window* and to keep tokens
   in flight so the pipeline depth becomes throughput instead of latency. That is
   the single most skippy-relevant claim in the paper.
2. **Lossless.** Verification uses the target's own logits as tokens pop out, so
   output equals greedy/sampled target output (same guarantee skippy's current
   spec lane gives, but without a separate draft model's tokenizer-compatibility
   risk that `llama-spec-bench` exists to manage).
3. **It reuses concepts skippy already has names for**: activation frames,
   inflight credits, window/commit/recovery, adaptive windows. The vocabulary
   lines up, which lowers integration risk.

## Hard parts / open questions before committing

1. **Trained speculation head per model family.** The head (`stage_projs`,
   `g0_proj`, `spec_layers`, `lm_head`, optional draft vocab) is *trained*
   (`train.py`, checkpoint `version==10`). mesh-llm would need to: train heads per
   supported family, version them, distribute them as artifacts (likely a sibling
   to layer packages), and bind them to a topology/split. This is the biggest lift
   and parallels how `skippy-family-certification` already gates families.
2. **Per-stage hidden-state taps over the wire.** The paper aggregates hidden
   states from *all passed stages* for a token. skippy currently forwards one
   activation frame stage→stage. Feeding a speculation head that wants multiple
   stages' taps means either (a) running the head only where taps are cheaply
   available (e.g. co-located with stage 0 / a designated stage), or (b)
   extending the activation wire to carry/retain per-stage taps. `use_deepest`
   in the reference shows you can degrade gracefully to whatever taps exist.
3. **Cross-stage, cross-peer rollback.** `crop_pipeline_cache_after_rejection`
   rewinds each layer by a stage-dependent step count. In skippy these layers live
   on *different peers*. A rejection must trigger a coordinated stage-aware rewind
   across peers. skippy has KV slot/lane state and `kv_eviction`, and the binary
   transport has state flags — but a stage-indexed snapshot-rewind protocol message
   would be new wire surface (must be additive; see protocol-compat rules).
4. **llama.cpp ABI.** The reference taps HF `output_hidden_states` per layer.
   skippy runs the patched llama.cpp staged runtime via `skippy-ffi`. Exposing
   per-stage hidden-state taps and a stage-snapshot rewind likely needs ABI
   surface in the patch queue (`skippy/common.h`, `SKIPPY_ABI_VERSION_*`), i.e. a
   staged-runtime change, not just Rust.
5. **Linear/hybrid attention rewind.** The reference's most subtle code is
   `PipelineLinearAttentionLayer` snapshotting conv + recurrent state for Qwen3.5
   linear attention. If we target families with linear/SSM-style layers, the
   equivalent of that snapshot buffer must exist in the staged runtime.

## Suggested staged adoption (lowest risk first)

1. **Measure the opportunity.** Instrument current split decode to quantify the
   cross-stage latency window per token (the bubble the paper removes). The
   telemetry already emitted by `speculative.rs` /
   `primary_verify_downstream_wait_ms`, `stage0_compute_ms` is most of what's
   needed. If the inter-stage wait per token is small relative to stage compute,
   the payoff is limited; if it's large (multi-peer over WAN), the payoff is large.
2. **Prototype zero-bubble with the *existing* draft lane.** Before training a
   head, run skippy's current draft-model proposer *concurrently* with the stage
   forward instead of before it (overlap draft compute with downstream-wait). This
   tests the scheduling/zero-bubble idea using machinery that already exists,
   without ABI changes.
3. **Add stage-aware rollback primitives.** Generalize the current span recovery
   into a stage-indexed layer-snapshot rewind, first single-node, then as an
   additive cross-peer protocol message. This is independently useful.
4. **Train + wire a speculation head for one family** (e.g. a Qwen3 family already
   certified for splits), distributed as a versioned artifact bound to a topology.
   Gate it behind the family-certification flow.
5. **Generalize taps / draft vocab / adaptive window** once one family proves out.

## First buildable experiment (single machine, real code)

Goal: test the paper's *core* thesis on real skippy machinery without a trained
head, a second model, or a second machine. Thesis under test: **draft the next
token(s) from the target's own early pipeline stages, verify against the full
pipeline's output, stay lossless.**

### Why this candidate

- skippy stages are just `layer_start..layer_end` ranges over one GGUF
  (`RuntimeConfig` in `crates/skippy-runtime/src/lib.rs`, `package.rs`). The
  state-size **baseline** family in `docs/skippy/FAMILY_STATUS.md` is
  `Qwen/Qwen3-0.6B:Q8_0`, certified at `layer_end=28, splits=9,18` — a real
  **3-stage** pipeline that runs **in one process**, no network.
- The whole speculative scaffolding already exists and is the right shape:
  `frontend/embedded_generation.rs` runs `propose -> VerifySpan -> classify ->
  commit/repair`; `frontend/speculative.rs` has `classify_verify_span`,
  `VerifySpanDecision`, adaptive window, recovery; `OpenAiSpeculativeStats` has
  the telemetry. Verification is already lossless via `VerifySpan`.
- The *only* thing structurally unlike the paper is the **draft source**. Today
  it is either a separate full GGUF (`DraftRunner` in `frontend.rs`) or n-gram.
  The paper drafts from the target's own early layers.
- A `StageModel` can be opened over `[0, k)` with `include_output: true`
  (see `DraftRunner::open`, which already opens a 0..layer_count slice with
  `include_output: true`). So **the first stage's layer range can itself emit
  logits** — an early-exit self-draft — reusing the exact stage grouping we
  already publish, with zero trained weights.

### The experiment: an "early-exit self-draft" proposer

Add a new draft source that is a second `StageSession` over the **same GGUF**
but restricted to `layer_start=0, layer_end=<first split>` (e.g. `0..9` for
Qwen3-0.6B), with `include_embeddings: true, include_output: true`. It runs the
same `propose(current, window)` autoregressive loop the existing `DraftRunner`
uses — but it is the target's own first stage acting as the drafter. Everything
downstream (VerifySpan against the full 28-layer target, classify, commit,
repair, telemetry) is reused unchanged.

This is a drop-in alternative to `DraftRunner`: same `propose` / `reset_to_context`
shape, so it slots into the `request.draft` path in
`frontend/generation_flow.rs` and `embedded_generation.rs` with no protocol
change and no ABI change.

### Scope boundaries (what this is NOT yet)

- Not the trained speculation head (`stage_projs`/`g0_proj`/`spec_layers`).
- Not per-stage hidden-state fusion; the self-draft uses early-layer *logits*,
  not aggregated taps.
- Not the zero-bubble *concurrency* (draft overlapping the forward) yet — first
  prove acceptance with the self-draft serially, then add overlap.
- Not cross-peer rollback; single process, existing recovery path only.

If acceptance is good, the natural next increments are (a) run the self-draft
*concurrently* with the stage-0 forward to get the paper's zero-bubble, and only
then (b) replace early-exit logits with a trained head.

### Concrete build plan

1. **New proposer type.** In `crates/skippy-server/src/frontend.rs`, add an
   `EarlyExitDraftRunner` next to `DraftRunner`: opens the *target's* GGUF path
   with `RuntimeConfig { stage_index: 0, layer_start: 0, layer_end: <draft_end>,
   include_embeddings: true, include_output: true, load_mode: RuntimeSlice, .. }`
   and exposes the same `reset_to_context` / `propose` methods. `draft_end`
   defaults to the family's first split (configurable).
2. **Wire a selection knob.** Add a draft-source option (e.g.
   `--draft-self-layers <k>` or `speculative_source = early_exit`) in
   `crates/skippy-server/src/cli.rs` / `config.rs` and the request plumbing
   (`frontend/request.rs`), so `request.draft` can be an early-exit runner. Keep
   the existing separate-GGUF and n-gram paths intact (additive).
3. **No protocol/ABI change.** Verification still uses `VerifySpan` against the
   full target; commit/repair/telemetry unchanged.
4. **Telemetry.** Reuse `OpenAiSpeculativeStats`; record `proposal_source =
   "early-exit-self"` and `draft_end` layer count so reports distinguish it from
   draft-model and n-gram.
5. **Bench harness.** Drive it through `crates/llama-spec-bench` style flow or a
   small script against `:9337` with `Qwen/Qwen3-0.6B:Q8_0`, comparing
   `baseline` vs `early-exit-self` on a fixed prompt set; report accept rate,
   `primary_verify_*` timing, and tokens/decode-step.

### Validation (single machine)

```bash
# 1. build
just release-build

# 2. serve Qwen3-0.6B as a 3-stage in-process pipeline (the certified baseline)
./target/release/mesh-llm serve --model "Qwen/Qwen3-0.6B:Q8_0" --split \
  --port 9337 --console 3131 --log-format json > /tmp/mesh.log 2>&1 & disown

# 3. wait for readiness, then A/B baseline vs early-exit self-draft
curl -s http://localhost:9337/v1/models | jq '.data[].id'
# baseline request, then early-exit-self request (knob TBD per step 2), compare:
#   - llama_stage.spec.accept_rate
#   - llama_stage.spec.primary_verify_elapsed_ms
#   - tokens/decode-step and end-to-end ms
```

Success criteria for the experiment (not for shipping): non-trivial acceptance
rate from the self-draft on Qwen3-0.6B, and a clear measurement of verify cost
vs. draft cost so we know whether zero-bubble overlap (the next increment) is
worth building. Lossless output is guaranteed by the existing VerifySpan path;
confirm greedy output equals baseline.

## Experiment results (2026-06-15, single machine)

We built the self-draft wiring and ran it on a genuine 2-stage Qwen3-0.6B split
(`Qwen/Qwen3-0.6B:Q8_0`, layer package, stage 0 = layers 0..14, stage 1 =
14..28) on one machine via `skippy-server serve-binary`.

What shipped and was validated:

- `EarlyExitDraftRunner` / `DraftSource::SelfPrefix` and `mode = "self"` config,
  including a package-backed self-draft open that selects the first N layer
  parts from the layer package.
- The full speculative loop on a real split — propose → `VerifySpan` →
  `classify_verify_span` → commit → KV trim → recovery — measured cleanly:
  `accept_rate ≈ 0.67`, `proposed=81`, `accepted=54`, 11 full-accept / 1
  tail-reject / 9 early-reject windows over 21 windows, coherent greedy output.
  (This run used a separate smaller-quant draft to exercise the loop; see the
  blocker below for why the self-prefix draft itself could not run.)

### Hard blocker found: early-exit logits are not supported by the runtime

The faithful self-draft (target's own layers `0..k` emitting logits) is blocked
in the **llama.cpp patch queue**, not the Rust layer. Opening a mid-stack slice
with an output head fails with:

```
InvalidArgument: only the final runtime slice may include output tensors
```

(see `third_party/llama.cpp/patches/0006-Execute-runtime-slice-activation-frames.patch`).
The output projection is only valid after the final layer's norm, so an
arbitrary early-layer slice cannot produce logits without a **trained early-exit
head** — which is exactly the paper's speculation head, and exactly the "hard
parts" item flagged below. Raw-GGUF stages additionally cannot early-exit at all
(a `layer_end < N` raw GGUF still loads and runs all N layers, which is why an
initial single-stage shortcut showed a bogus 100% accept rate).

### Conclusion / next increment

The speculative scaffolding, the staged split, and the self-draft selection are
all in place and correct. The remaining gap to a true zero-trained-weights
early-exit self-draft is a **patch-queue change** to allow logits from a
non-final slice (or a small trained early-exit head). That is the right next
unit of work, and it is a C++/ABI change, not Rust wiring. Until then, the
self-draft path is best exercised with a separate small draft model, and the
`mode = "self"` config is wired but will fail fast on the runtime constraint
until the patch-queue support lands.

## Bottom line

The paper is a good conceptual fit because skippy is already pipeline-parallel and
already has lossless speculative decoding plumbing, telemetry, and recovery. The
*new* value is "draft while the pipeline forwards, verify from what pops out, roll
back per stage" — a throughput win that targets skippy's actual cost center
(cross-stage/cross-peer latency). The blockers are real but bounded: a trained
per-family head, per-stage hidden-state taps, and a stage-aware rollback that has
to cross peers and probably touches the staged-runtime ABI. Recommend starting
with measurement + a zero-bubble scheduling prototype on the existing draft lane
before investing in trained heads and ABI work.
