# Wiring `skippy-engine-mlx` into mesh-llm

This crate is a **working, self-contained MLX (Metal) serving engine** that
already serves HF safetensors models over mesh-llm's real OpenAI frontend
(`openai_frontend::router_for`). It is intentionally standalone right now — its
own cargo workspace, path-depending the local safemlx fork — so it does not
perturb the main workspace or CI. This document is the concrete plan to promote
it into the shipped binary so that **on a Mac, `mesh-llm serve` can run an MLX
tensor model and users can pick one from `/v1/models`.**

## What already works (this crate, today)

- `MlxEngine` — a dedicated OS worker thread owns the non-`Send` MLX objects
  (model, streams, arrays); the outside world talks to it with `Send` channels.
- `MlxBackend: openai_frontend::OpenAiBackend` — `models`, `chat_completion`,
  `chat_completion_stream` (SSE), with usage accounting and incremental
  detokenization.
- `mlx-serve` bin — `router_for(Arc<MlxBackend>)` + `axum::serve`.
- Verified on Apple Silicon (Metal): `/v1/models`, non-stream chat, and stream
  chat all return real Qwen3-0.6B generations. Source precision ~321 tok/s;
  JIT-4bit ~604 tok/s (see `../../spikes/mlx-solo/FINDINGS.md`).

## Promotion plan (the actual PR)

### 1. Make it a real workspace member with a pinned safemlx

- Add `crates/skippy-engine-mlx` to root `Cargo.toml` `members` and remove its
  local `[workspace]` table.
- Replace the `path = "../../../safemlx/..."` deps with **git-rev pins** of
  `jbg/safemlx` (published crates are broken for Qwen3 — see FINDINGS §"supply
  chain"). Carry the two small loader fixes as a patch/branch until upstreamed.
- Keep the crate's `mlx` feature; gate all MLX code with
  `#[cfg(all(feature = "mlx", target_os = "macos"))]` (already done).
- Update `scripts/affected-crates.sh`, `scripts/plan-clippy-batches.sh`,
  `scripts/publish-crates.sh` `WORKSPACE_MEMBERS`, and run
  `cargo run -p xtask -- repo-consistency ci-crate-lists`. Because MLX is a heavy
  native lane, add it to the CI backend-gating like the other native features
  (only build/test the `mlx` feature on macOS runners).

### 2. Depend on it from host-runtime, macOS-gated

In `crates/mesh-llm-host-runtime/Cargo.toml`:

```toml
[target.'cfg(target_os = "macos")'.dependencies]
skippy-engine-mlx = { path = "../skippy-engine-mlx", optional = true }

[features]
mlx = ["dep:skippy-engine-mlx", "skippy-engine-mlx/mlx"]
```

Propagate a `mlx` feature up through `crates/mesh-llm/Cargo.toml`, and enable it
by default only on macOS builds in the release packaging.

### 3. Add an `Mlx` variant to the launch enum

`crates/mesh-llm-host-runtime/src/runtime/local.rs`:

- `LocalRuntimeBackendHandle` currently has one variant (`Skippy { .. }`). Add
  (macOS+feature gated):

  ```rust
  #[cfg(all(feature = "mlx", target_os = "macos"))]
  Mlx { backend: Arc<skippy_engine_mlx::MlxBackend>, http: MlxHttpHandle, _death_tx: ... },
  ```

- Every `match &self.inner { LocalRuntimeBackendHandle::Skippy { .. } => ... }`
  in `local.rs` (pid, ctx_used_tokens, openai_guardrails, llama_slots_snapshot,
  set_openai_guardrail_mode, shutdown/http accessor) needs an `Mlx` arm. Most map
  to simple/None-ish values since MLX has no llama slots or GGUF guardrail state.

### 4. Route safetensors models to the MLX branch at launch

`start_runtime_local_model` currently branches:

```
if is_layer_package_ref(..) { layer_package } else { skippy (direct GGUF) }
```

Add a first branch: if the resolved model is `ModelFormat::Safetensors` (the
model layer already classifies this — `crates/model-artifact` `ModelFormat`, and
`models/resolve` already detects `is_primary_mlx_weight_file`), and we're on
macOS with the `mlx` feature, start an `MlxEngine` instead:

```rust
#[cfg(all(feature = "mlx", target_os = "macos"))]
if resolved_format == ModelFormat::Safetensors {
    return start_runtime_mlx_model(spec, model_name, plan).await;
}
```

`start_runtime_mlx_model` mirrors `start_runtime_skippy_model`: build an
`MlxEngineConfig` from the resolved model dir + planned ctx/limits, `spawn` the
engine on a blocking task, wrap `MlxBackend` in the embedded HTTP handle
(`openai_frontend::router_for`), and return a `LocalRuntimeModelHandle` whose
`backend` string is `"mlx"`.

### 5. Model discovery / listing already works

The model layer already discovers, downloads, catalogs, and lists MLX
safetensors repos (`crates/mesh-llm-host-runtime/src/models/catalog.rs`,
`.../models/resolve`, `crates/model-resolver`). No change needed for a user to
*see* MLX models; the missing piece was purely the serving engine, which this
crate provides. Auto-behavior: on a Mac, a resolved safetensors model simply
routes to the MLX engine.

### 6. Auto on Mac + user selection

- With the `mlx` feature enabled by default on macOS builds, serving a
  safetensors model "just works" with no extra flags.
- Users pick a model the same way as today: `mesh-llm serve --model <hf-repo>`
  or from `~/.mesh-llm/config.toml`; safetensors → MLX, GGUF → llama.cpp.
- Optionally add `--serving-backend mlx|llama` to force the engine when a model
  is available in both formats (parallels the existing skippy backend selector
  noted in `docs/SKIPPY.md`).

## Out of scope for the first PR

- **Splits / staged execution.** This is single-stage, whole-model serving. The
  staged `StageEngine` trait, partial-load, and activation-frame work (plan §6–§8)
  are separate and remain gated on the go/no-go spikes.
- **Tool calling / reasoning parsing.** goose's `mlx.rs` has native + emulated
  tool parsing and thinking-output filtering worth porting later; this crate
  streams raw model text (including `<think>` blocks) for now.
- **Draft/speculative decoding** (goose's `gemma4_mtp`).

## Testing the promoted path

- `just build` on macOS with the `mlx` feature.
- `mesh-llm serve --model <mlx-safetensors-repo>` → confirm the model appears in
  `/v1/models` and `/v1/chat/completions` returns a generation.
- Confirm non-macOS / no-feature builds are byte-for-byte unaffected (the crate
  and its deps compile out entirely).
